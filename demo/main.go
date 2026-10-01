// Local Kimodo text-to-motion demo. Generation is serialized so one native
// process owns Vulkan at a time, while the persistent gallery stays readable.
package main

import (
	"bufio"
	"crypto/rand"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed index.html
var files embed.FS

//go:embed comparison.html
var comparisonPage []byte

//go:embed models.js
var modelUI []byte

//go:embed assets/localai.png
var localAILogo []byte

type animation struct {
	ID               string          `json:"id"`
	Prompt           string          `json:"prompt"`
	Frames           int             `json:"frames"`
	DiffusionSteps   int             `json:"diffusion_steps"`
	Seed             uint64          `json:"seed"`
	CreatedAt        string          `json:"created_at"`
	Status           string          `json:"status"`
	Error            string          `json:"error,omitempty"`
	Kind             string          `json:"kind"`
	Model            string          `json:"model"`
	TextQuantization string          `json:"text_quantization"`
	Segments         []promptSegment `json:"segments,omitempty"`
	TransitionFrames int             `json:"transition_frames,omitempty"`
	Progress         string          `json:"progress,omitempty"`
}
type promptSegment struct {
	Prompt string `json:"prompt"`
	Frames int    `json:"frames"`
}
type motionModel struct {
	ID          string       `json:"id"`
	Label       string       `json:"label"`
	Skeleton    string       `json:"skeleton"`
	SkeletonKey string       `json:"skeleton_key"`
	Upstream    string       `json:"upstream"`
	License     string       `json:"license"`
	LicenseURL  string       `json:"license_url"`
	Commercial  bool         `json:"commercial"`
	Available   bool         `json:"available"`
	Reason      string       `json:"reason,omitempty"`
	Parents     []int        `json:"parents"`
	Offsets     [][3]float32 `json:"offsets"`
	Motion      string       `json:"-"`
}
type textBundle struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Description string `json:"description"`
	Available   bool   `json:"available"`
	Reason      string `json:"reason,omitempty"`
	Bytes       int64  `json:"bytes,omitempty"`
	Path        string `json:"-"`
}
type gallery struct {
	mu                sync.RWMutex
	items             map[string]*animation
	output, generator string
	queue             chan string
	models            map[string]motionModel
	textBundles       map[string]textBundle
}

type generatorSession struct {
	key    string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

// envOr keeps the built-in default unless the deployment (e.g. a
// container) pre-sets the variable; appended after os.Environ() below,
// an explicit value would otherwise be overridden by the hardcode.
func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func startGeneratorSession(generator string, model motionModel, text textBundle) (*generatorSession, error) {
	cmd := exec.Command(generator, "--server", model.Motion, text.Path)
	// Deployments may force KIMODO_BACKEND=cpu; the vulkan default itself
	// falls back to CPU when no device is present.  The memory knobs are
	// likewise overridable so small-VRAM hosts can bound residency.
	cmd.Env = append(os.Environ(),
		"KIMODO_BACKEND="+envOr("KIMODO_BACKEND", "vulkan"),
		"KIMODO_TEXT_LAYER_CHUNK="+envOr("KIMODO_TEXT_LAYER_CHUNK", "32"),
		"KIMODO_TEXT_RESIDENT_LIMIT_MIB="+envOr("KIMODO_TEXT_RESIDENT_LIMIT_MIB", "10000"))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}
	return &generatorSession{key: model.Motion + "\x00" + text.Path, cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}, nil
}

func (session *generatorSession) close() {
	if session == nil {
		return
	}
	_ = session.stdin.Close()
	_ = session.cmd.Wait()
}

func (session *generatorSession) generate(item *animation, dir string, segments []promptSegment, promptPaths []string) error {
	fields := []string{fmt.Sprint(item.TransitionFrames), fmt.Sprint(item.DiffusionSteps), fmt.Sprint(item.Seed), dir}
	for index, segment := range segments {
		fields = append(fields, fmt.Sprint(segment.Frames), promptPaths[index])
	}
	if _, err := fmt.Fprintln(session.stdin, strings.Join(fields, "\t")); err != nil {
		return err
	}
	response, err := session.stdout.ReadString('\n')
	if err != nil {
		return fmt.Errorf("native worker stopped: %w", err)
	}
	response = strings.TrimSpace(response)
	if strings.HasPrefix(response, "ERR\t") {
		return fmt.Errorf("native worker: %s", strings.TrimPrefix(response, "ERR\t"))
	}
	if !strings.HasPrefix(response, "OK\t") {
		return fmt.Errorf("invalid native worker response %q", response)
	}
	return nil
}

func token() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func safePathPart(value string) bool {
	if value == "" || value == "." || value == ".." {
		return false
	}
	for _, r := range value {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.') {
			return false
		}
	}
	return true
}
func (g *gallery) save(a *animation) error {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(g.output, a.ID+".json"), b, 0644)
}
func (g *gallery) list() []*animation {
	g.mu.RLock()
	defer g.mu.RUnlock()
	result := make([]*animation, 0, len(g.items))
	for _, item := range g.items {
		copy := *item
		result = append(result, &copy)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].CreatedAt > result[j].CreatedAt })
	return result
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("invalid F32 file: %s", path)
	}
	values := make([]float32, len(b)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return values, nil
}

func writeF32(path string, values []float32) error {
	b := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(value))
	}
	return os.WriteFile(path, b, 0600)
}

// Each motion is exported as a node-only GLB: it deliberately has no mesh or
// skin, so consumers can attach their own Three.js geometry to the named
// joints. Kimodo stores root translations and local XYZW rotations.
var smplx22Parents = [...]int{-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19}
var smplx22Names = [...]string{"pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
var smplx22Offsets = [...][3]float32{{}, {.052299, -.093936, -.027607}, {-.057193, -.106548, -.022218}, {-.001496, .11293, -.024981}, {.058867, -.416442, -.006557}, {-.048074, -.39756, -.014061}, {.0069, .145636, -.006859}, {-.041738, -.437584, -.029512}, {.014489, -.446853, -.01803}, {-.010334, .056082, .021116}, {.049294, -.065279, .126259}, {-.040575, -.065287, .127076}, {-.011026, .171365, -.028827}, {.047725, .087643, -.008375}, {-.046636, .086612, -.014864}, {.024654, .175391, .024463}, {.126285, .05768, -.013885}, {-.109342, .053674, -.009118}, {.272907, -.069853, -.039094}, {-.292029, -.03544, -.024565}, {.276174, .021254, -.002478}, {-.271878, -.004835, -.016445}}

type skeletonDefinition struct {
	key     string
	names   []string
	parents []int
	offsets [][3]float32
}

var skeletonDefinitions = map[string]skeletonDefinition{
	"smplx22": {key: "smplx22", names: smplx22Names[:], parents: smplx22Parents[:], offsets: smplx22Offsets[:]},
}

type gltfBufferView struct {
	Buffer     int `json:"buffer"`
	ByteOffset int `json:"byteOffset,omitempty"`
	ByteLength int `json:"byteLength"`
}
type gltfAccessor struct {
	BufferView    int    `json:"bufferView"`
	ComponentType int    `json:"componentType"`
	Count         int    `json:"count"`
	Type          string `json:"type"`
}

func appendF32(dst []byte, values []float32) []byte {
	for _, value := range values {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(value))
		dst = append(dst, b[:]...)
	}
	return dst
}

func writeSkeletonGLB(path string, roots, rotations []float32, skeleton skeletonDefinition) error {
	frames := len(roots) / 3
	joints := len(skeleton.parents)
	if frames < 1 || joints < 1 || len(skeleton.names) != joints || len(skeleton.offsets) != joints || len(roots) != frames*3 || len(rotations) != frames*joints*4 {
		return fmt.Errorf("invalid %s motion for GLB export", skeleton.key)
	}
	times := make([]float32, frames)
	for i := range times {
		times[i] = float32(i) / 30
	}
	bin := make([]byte, 0, (frames+frames*3+frames*22*4)*4)
	views := make([]gltfBufferView, 0, 24)
	addView := func(values []float32) int {
		offset := len(bin)
		bin = appendF32(bin, values)
		views = append(views, gltfBufferView{Buffer: 0, ByteOffset: offset, ByteLength: len(bin) - offset})
		return len(views) - 1
	}
	timeView, rootView := addView(times), addView(roots)
	rotationViews := make([]int, joints)
	for joint := range rotationViews {
		track := make([]float32, frames*4)
		for frame := 0; frame < frames; frame++ {
			copy(track[frame*4:], rotations[(frame*joints+joint)*4:(frame*joints+joint+1)*4])
		}
		rotationViews[joint] = addView(track)
	}
	accessors := []gltfAccessor{{BufferView: timeView, ComponentType: 5126, Count: frames, Type: "SCALAR"}, {BufferView: rootView, ComponentType: 5126, Count: frames, Type: "VEC3"}}
	for _, view := range rotationViews {
		accessors = append(accessors, gltfAccessor{BufferView: view, ComponentType: 5126, Count: frames, Type: "VEC4"})
	}
	nodes := make([]map[string]any, joints)
	for joint := range nodes {
		node := map[string]any{"name": skeleton.names[joint]}
		if joint != 0 {
			node["translation"] = skeleton.offsets[joint]
		}
		children := make([]int, 0, 3)
		for child, parent := range skeleton.parents {
			if parent == joint {
				children = append(children, child)
			}
		}
		if len(children) != 0 {
			node["children"] = children
		}
		nodes[joint] = node
	}
	samplers := make([]map[string]any, 0, 23)
	channels := make([]map[string]any, 0, 23)
	addChannel := func(node, output int, path string) {
		samplers = append(samplers, map[string]any{"input": 0, "output": output, "interpolation": "LINEAR"})
		channels = append(channels, map[string]any{"sampler": len(samplers) - 1, "target": map[string]any{"node": node, "path": path}})
	}
	addChannel(0, 1, "translation")
	for joint := 0; joint < joints; joint++ {
		addChannel(joint, joint+2, "rotation")
	}
	document := map[string]any{
		"asset":       map[string]string{"version": "2.0", "generator": "kimodo.cpp skeleton exporter"},
		"scene":       0,
		"scenes":      []map[string]any{{"nodes": []int{0}}},
		"nodes":       nodes,
		"buffers":     []map[string]int{{"byteLength": len(bin)}},
		"bufferViews": views,
		"accessors":   accessors,
		"animations":  []map[string]any{{"name": "KimodoMotion", "samplers": samplers, "channels": channels}},
		"extras":      map[string]any{"skeleton": skeleton.key, "fps": 30, "rotation_order": "xyzw"},
	}
	jsonChunk, err := json.Marshal(document)
	if err != nil {
		return err
	}
	for len(jsonChunk)%4 != 0 {
		jsonChunk = append(jsonChunk, ' ')
	}
	for len(bin)%4 != 0 {
		bin = append(bin, 0)
	}
	total := 12 + 8 + len(jsonChunk) + 8 + len(bin)
	out := make([]byte, 0, total)
	putU32 := func(value uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], value)
		out = append(out, b[:]...)
	}
	putU32(0x46546c67)
	putU32(2)
	putU32(uint32(total))
	putU32(uint32(len(jsonChunk)))
	putU32(0x4e4f534a)
	out = append(out, jsonChunk...)
	putU32(uint32(len(bin)))
	putU32(0x004e4942)
	out = append(out, bin...)
	return os.WriteFile(path, out, 0600)
}

func exportSkeletonGLB(dir, skeletonKey string) error {
	skeleton, ok := skeletonDefinitions[skeletonKey]
	if !ok {
		return fmt.Errorf("unsupported skeleton %q", skeletonKey)
	}
	roots, err := readF32(filepath.Join(dir, "root_positions.f32"))
	if err != nil {
		return err
	}
	rotations, err := readF32(filepath.Join(dir, "local_rotations_xyzw.f32"))
	if err != nil {
		return err
	}
	return writeSkeletonGLB(filepath.Join(dir, "animation.glb"), roots, rotations, skeleton)
}

func (g *gallery) worker() {
	var session *generatorSession
	defer func() { session.close() }()
	for id := range g.queue {
		g.mu.Lock()
		item := g.items[id]
		item.Status = "running"
		_ = g.save(item)
		g.mu.Unlock()
		dir := filepath.Join(g.output, id)
		err := os.MkdirAll(dir, 0755)
		if err == nil {
			err = os.WriteFile(filepath.Join(dir, "prompt.txt"), []byte(item.Prompt), 0600)
		}
		if err == nil {
			model, ok := g.models[item.Model]
			if !ok || !model.Available {
				err = fmt.Errorf("model %q is not available", item.Model)
			} else {
				textID := item.TextQuantization
				if textID == "" {
					textID = "q8_0"
				}
				text, textOK := g.textBundles[textID]
				if !textOK || !text.Available {
					err = fmt.Errorf("text quantization %q is not available", textID)
				}
				if err == nil {
					segments := item.Segments
					if len(segments) == 0 {
						segments = []promptSegment{{Prompt: item.Prompt, Frames: item.Frames}}
					}
					promptPaths := make([]string, 0, len(segments))
					for index, segment := range segments {
						promptPath := filepath.Join(dir, fmt.Sprintf("segment-%02d.txt", index+1))
						if err = os.WriteFile(promptPath, []byte(segment.Prompt), 0600); err != nil {
							break
						}
						promptPaths = append(promptPaths, promptPath)
					}
					if err == nil {
						g.mu.Lock()
						item.Progress = fmt.Sprintf("Generating %d conditioned segments", len(segments))
						_ = g.save(item)
						g.mu.Unlock()
						key := model.Motion + "\x00" + text.Path
						if session == nil || session.key != key {
							session.close()
							session, err = startGeneratorSession(g.generator, model, text)
						}
						if err == nil {
							err = session.generate(item, dir, segments, promptPaths)
							if err != nil {
								session.close()
								session = nil
							}
						}
					}
					if err == nil {
						err = exportSkeletonGLB(dir, model.SkeletonKey)
					}
				}
			}
		}
		g.mu.Lock()
		if err != nil {
			item.Status = "failed"
			item.Error = err.Error()
		} else {
			item.Status = "ready"
			item.Progress = ""
		}
		if saveErr := g.save(item); saveErr != nil {
			log.Printf("save %s: %v", item.ID, saveErr)
		}
		g.mu.Unlock()
	}
}

func main() {
	preferPacked := func(packed, legacy string) string {
		if info, err := os.Stat(packed); err == nil && info.Mode().IsRegular() {
			return packed
		}
		return legacy
	}
	addr := flag.String("addr", "127.0.0.1:8090", "listen address")
	motion := flag.String("motion-model", "models/kimodo-smplx-rp-v1-f32.gguf", "motion GGUF")
	somaRP := flag.String("soma-rp-model", "models/kimodo-soma-rp-v1.1-f32.gguf", "SOMA RP v1.1 motion GGUF")
	somaSEED := flag.String("soma-seed-model", "models/kimodo-soma-seed-v1.1-f32.gguf", "SOMA SEED v1.1 motion GGUF")
	g1RP := flag.String("g1-rp-model", "models/kimodo-g1-rp-v1-f32.gguf", "G1 RP v1 motion GGUF")
	g1SEED := flag.String("g1-seed-model", "models/kimodo-g1-seed-v1-f32.gguf", "G1 SEED v1 motion GGUF")
	text := flag.String("text-bundle", preferPacked("Llama-3-Kimodo-BF16.gguf", "generated/llm2vec-text-bundle"), "BF16 LLM2Vec GGUF or legacy component directory")
	textQ8 := flag.String("text-q8-bundle", preferPacked("Llama-3-Kimodo-Q8_0.gguf", "generated/llm2vec-text-q8_0"), "Q8_0 LLM2Vec GGUF or legacy component directory")
	textQ6 := flag.String("text-q6-bundle", preferPacked("Llama-3-Kimodo-Q6_K.gguf", "generated/llm2vec-text-q6_k"), "Q6_K LLM2Vec GGUF or legacy component directory")
	textQ5 := flag.String("text-q5-bundle", preferPacked("Llama-3-Kimodo-Q5_K.gguf", "generated/llm2vec-text-q5_k"), "Q5_K LLM2Vec GGUF or legacy component directory")
	textQ4 := flag.String("text-q4-bundle", preferPacked("Llama-3-Kimodo-Q4_K.gguf", "generated/llm2vec-text-q4_k"), "Q4_K LLM2Vec GGUF or legacy component directory")
	textQ4Mixed := flag.String("text-q4-mixed-bundle", preferPacked("Llama-3-Kimodo-Q4_K_M.gguf", "generated/llm2vec-text-q4_k_m"), "mixed Q4_K LLM2Vec GGUF or legacy component directory")
	generator := flag.String("generator", "build/debug/kmd-generate", "native text-to-motion command")
	output := flag.String("output", "demo-output", "persistent gallery directory")
	comparisons := flag.String("comparisons", "quantization-comparisons", "viewer-ready quantization comparison directories")
	flag.Parse()
	if err := os.MkdirAll(*output, 0755); err != nil {
		log.Fatal(err)
	}
	makeModel := func(id, label, skeletonLabel, skeletonKey, upstream, license, licenseURL, path string, commercial bool) motionModel {
		definition := skeletonDefinitions[skeletonKey]
		model := motionModel{ID: id, Label: label, Skeleton: skeletonLabel, SkeletonKey: skeletonKey, Upstream: upstream, License: license, LicenseURL: licenseURL, Commercial: commercial, Parents: definition.parents, Offsets: definition.offsets, Motion: path}
		if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
			model.Available = true
		} else {
			model.Reason = "GGUF not found at " + path
		}
		return model
	}
	const internalLicense = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/"
	const openLicense = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
	models := map[string]motionModel{
		"smplx-rp-v1":    makeModel("smplx-rp-v1", "SMPL-X RP v1", "SMPL-X 22 joints", "smplx22", "nvidia/Kimodo-SMPLX-RP-v1", "NVIDIA Internal Scientific R&D (non-commercial)", internalLicense, *motion, false),
		"soma-rp-v1.1":   makeModel("soma-rp-v1.1", "SOMA RP v1.1", "SOMA compact 30-joint control skeleton", "soma30", "nvidia/Kimodo-SOMA-RP-v1.1", "NVIDIA Open Model License", openLicense, *somaRP, true),
		"soma-seed-v1.1": makeModel("soma-seed-v1.1", "SOMA SEED v1.1", "SOMA compact 30-joint control skeleton", "soma30", "nvidia/Kimodo-SOMA-SEED-v1.1", "NVIDIA Open Model License", openLicense, *somaSEED, true),
		"g1-rp-v1":       makeModel("g1-rp-v1", "G1 RP v1", "Unitree G1 34 joints", "g1skel34", "nvidia/Kimodo-G1-RP-v1", "NVIDIA Open Model License", openLicense, *g1RP, true),
		"g1-seed-v1":     makeModel("g1-seed-v1", "G1 SEED v1", "Unitree G1 34 joints", "g1skel34", "nvidia/Kimodo-G1-SEED-v1", "NVIDIA Open Model License", openLicense, *g1SEED, true),
	}
	makeTextBundle := func(id, label, description, path string) textBundle {
		bundle := textBundle{ID: id, Label: label, Description: description, Path: path}
		info, err := os.Stat(path)
		if err != nil {
			bundle.Reason = "bundle not found at " + path
			return bundle
		}
		components := []string{path}
		if info.IsDir() {
			components, err = filepath.Glob(filepath.Join(path, "*.gguf"))
			if err != nil || len(components) == 0 {
				bundle.Reason = "bundle contains no GGUF components"
				return bundle
			}
		} else if info.Mode().IsRegular() && filepath.Ext(path) == ".gguf" {
			tokenizer := filepath.Join(filepath.Dir(path), "tokenizer.gguf")
			if tokenizerInfo, statErr := os.Stat(tokenizer); statErr != nil || !tokenizerInfo.Mode().IsRegular() {
				bundle.Reason = "tokenizer.gguf not found beside " + path
				return bundle
			}
			components = append(components, tokenizer)
		} else {
			bundle.Reason = "text model is not a GGUF or component directory"
			return bundle
		}
		for _, component := range components {
			if componentInfo, statErr := os.Stat(component); statErr == nil {
				bundle.Bytes += componentInfo.Size()
			}
		}
		bundle.Available = true
		return bundle
	}
	textBundles := map[string]textBundle{
		"bf16":   makeTextBundle("bf16", "BF16 reference", "Highest-fidelity reference encoder.", *text),
		"q8_0":   makeTextBundle("q8_0", "Q8_0", "Recommended quantized encoder; closest measured agreement with BF16.", *textQ8),
		"q6_k":   makeTextBundle("q6_k", "Q6_K", "Smaller experimental encoder with increased motion divergence.", *textQ6),
		"q5_k":   makeTextBundle("q5_k", "Q5_K", "Experimental; some prompt/noise pairs enter a different trajectory basin.", *textQ5),
		"q4_k":   makeTextBundle("q4_k", "Q4_K", "Experimental uniform four-bit encoder with substantial measured divergence.", *textQ4),
		"q4_k_m": makeTextBundle("q4_k_m", "Q4_K mixed", "Experimental mixed-bit profile that improves on uniform Q4_K.", *textQ4Mixed),
	}
	g := &gallery{items: map[string]*animation{}, output: *output, queue: make(chan string, 32), generator: *generator, models: models, textBundles: textBundles}
	entries, _ := filepath.Glob(filepath.Join(*output, "*.json"))
	for _, path := range entries {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var a animation
		if json.Unmarshal(b, &a) == nil {
			if a.TextQuantization == "" {
				a.TextQuantization = "bf16"
			}
			g.items[a.ID] = &a
			if a.Status == "ready" {
				model, ok := models[a.Model]
				if !ok {
					model = models["smplx-rp-v1"]
				}
				if err := exportSkeletonGLB(filepath.Join(*output, a.ID), model.SkeletonKey); err != nil && !os.IsNotExist(err) {
					log.Printf("export existing animation %s: %v", a.ID, err)
				}
			}
		}
	}
	go g.worker()
	index, err := files.ReadFile("index.html")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(index)
	})
	mux.HandleFunc("/compare", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/compare" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(comparisonPage)
	})
	mux.HandleFunc("/api/comparisons", func(w http.ResponseWriter, r *http.Request) {
		entries, _ := os.ReadDir(*comparisons)
		result := make([]json.RawMessage, 0, len(entries))
		for _, entry := range entries {
			if !entry.IsDir() || !safePathPart(entry.Name()) {
				continue
			}
			data, err := os.ReadFile(filepath.Join(*comparisons, entry.Name(), "comparison.json"))
			if err == nil && json.Valid(data) {
				result = append(result, data)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	})
	mux.HandleFunc("/api/comparisons/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/comparisons/"), "/")
		if len(parts) < 2 || len(parts) > 3 || !safePathPart(parts[0]) {
			http.NotFound(w, r)
			return
		}
		var path string
		if len(parts) == 2 && parts[1] == "comparison.json" {
			path = filepath.Join(*comparisons, parts[0], "comparison.json")
			w.Header().Set("Content-Type", "application/json")
		} else if len(parts) == 3 && safePathPart(parts[1]) &&
			(parts[2] == "root_positions.f32" || parts[2] == "local_rotations_xyzw.f32") {
			path = filepath.Join(*comparisons, parts[0], parts[1], parts[2])
			w.Header().Set("Content-Type", "application/octet-stream")
		} else {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		http.ServeFile(w, r, path)
	})
	mux.HandleFunc("/localai.png", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Cache-Control", "public, max-age=86400")
		_, _ = w.Write(localAILogo)
	})
	mux.HandleFunc("/models.js", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(modelUI)
	})
	mux.HandleFunc("/api/animations", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(g.list())
	})
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		result := make([]motionModel, 0, len(g.models))
		for _, model := range g.models {
			result = append(result, model)
		}
		sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	})
	mux.HandleFunc("/api/text-quantizations", func(w http.ResponseWriter, r *http.Request) {
		order := []string{"bf16", "q8_0", "q6_k", "q5_k", "q4_k", "q4_k_m"}
		result := make([]textBundle, 0, len(order))
		for _, id := range order {
			result = append(result, g.textBundles[id])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	})
	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		var request struct {
			Prompt           string          `json:"prompt"`
			Segments         []promptSegment `json:"segments"`
			TransitionFrames int             `json:"transition_frames"`
			Frames           int             `json:"frames"`
			Steps            int             `json:"steps"`
			Seed             uint64          `json:"seed"`
			Model            string          `json:"model"`
			TextQuantization string          `json:"text_quantization"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10)).Decode(&request); err != nil {
			http.Error(w, "invalid JSON", 400)
			return
		}
		request.Prompt = strings.TrimSpace(request.Prompt)
		if len(request.Segments) == 0 {
			request.Segments = []promptSegment{{Prompt: request.Prompt, Frames: request.Frames}}
		}
		if len(request.Segments) > 16 {
			http.Error(w, "at most 16 prompt segments", 400)
			return
		}
		if request.Frames == 0 {
			request.Frames = 150
		}
		if request.Steps == 0 {
			request.Steps = 100
		}
		for index := range request.Segments {
			request.Segments[index].Prompt = strings.TrimSpace(request.Segments[index].Prompt)
			if request.Segments[index].Frames == 0 {
				request.Segments[index].Frames = 150
			}
			if request.Segments[index].Prompt == "" || len(request.Segments[index].Prompt) > 4096 || request.Segments[index].Frames < 60 || request.Segments[index].Frames > 150 {
				http.Error(w, "each prompt segment must be 60..150 frames and 1..4096 bytes", 400)
				return
			}
		}
		if request.TransitionFrames == 0 {
			request.TransitionFrames = 5
		}
		if request.TransitionFrames < 1 || request.TransitionFrames > 60 || request.Steps < 1 || request.Steps > 1000 {
			http.Error(w, "transition frames must be 1..60 and steps 1..1000", 400)
			return
		}
		if request.Model == "" {
			request.Model = "smplx-rp-v1"
		}
		if request.TextQuantization == "" {
			request.TextQuantization = "q8_0"
		}
		model, ok := g.models[request.Model]
		if !ok || !model.Available {
			http.Error(w, "selected motion model is not available: "+model.Reason, http.StatusConflict)
			return
		}
		text, ok := g.textBundles[request.TextQuantization]
		if !ok {
			http.Error(w, "unknown text quantization", http.StatusBadRequest)
			return
		}
		if !text.Available {
			http.Error(w, "selected text quantization is not available: "+text.Reason, http.StatusConflict)
			return
		}
		totalFrames := 0
		for _, segment := range request.Segments {
			totalFrames += segment.Frames
		}
		a := &animation{ID: token(), Prompt: request.Segments[0].Prompt, Frames: totalFrames, DiffusionSteps: request.Steps, Seed: request.Seed, CreatedAt: time.Now().UTC().Format(time.RFC3339), Status: "queued", Kind: "generated", Model: request.Model, TextQuantization: request.TextQuantization, Segments: request.Segments, TransitionFrames: request.TransitionFrames}
		g.mu.Lock()
		g.items[a.ID] = a
		err := g.save(a)
		g.mu.Unlock()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		g.queue <- a.ID
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(a)
	})
	mux.HandleFunc("/api/animations/", func(w http.ResponseWriter, r *http.Request) {
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/animations/"), "/")
		if len(parts) != 2 || (parts[1] != "root.f32" && parts[1] != "rotations.f32" && parts[1] != "animation.glb") {
			http.NotFound(w, r)
			return
		}
		g.mu.RLock()
		a := g.items[parts[0]]
		g.mu.RUnlock()
		if a == nil || a.Status != "ready" {
			http.NotFound(w, r)
			return
		}
		name := "root_positions.f32"
		if parts[1] == "rotations.f32" {
			name = "local_rotations_xyzw.f32"
		}
		if parts[1] == "animation.glb" {
			name = "animation.glb"
			w.Header().Set("Content-Type", "model/gltf-binary")
			w.Header().Set("Content-Disposition", "attachment; filename=kimodo-"+a.ID+".glb")
			// A GLB is a compact asset; read it directly so browsers always receive
			// it as a download rather than invoking any path-cleaning redirects.
			data, err := os.ReadFile(filepath.Join(g.output, a.ID, name))
			if err != nil {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			_, _ = w.Write(data)
			return
		} else {
			w.Header().Set("Content-Type", "application/octet-stream")
		}
		w.Header().Set("Cache-Control", "no-store")
		http.ServeFile(w, r, filepath.Join(g.output, a.ID, name))
	})
	log.Printf("Kimodo text-to-motion demo listening at http://%s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
