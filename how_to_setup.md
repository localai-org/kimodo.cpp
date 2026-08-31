# Kimodo.cpp: Complete Setup & Unreal Engine 5 Guide

A step-by-step guide to setting up **kimodo.cpp** on Windows, building the C++ binaries with **Visual Studio 2022**, generating text-to-motion animations, and applying them to the **Unreal Engine 5 Mannequin** (`SKM_Manny` / `SKM_Quinn`).

---

## 1. Prerequisites

Before starting, ensure you have the following installed on your system:
- **Visual Studio 2022** (with *Desktop development with C++* and *C++ CMake tools for Windows*).
- **Python 3.10+** (with `huggingface_hub` installed: `pip install huggingface_hub`).
- **Unreal Engine 5** (with the *Third Person Template*).
- **Git**.

---

## 2. Clone the Repository & Submodules

Open **PowerShell** and run:

```powershell
# 1. Clone the repository
git clone https://github.com/Deepesh70/kimodo.cpp.git E:\Kimodo
cd E:\Kimodo

# 2. Initialize and update the GGML submodule
git submodule update --init --recursive
```

---

## 3. Build with Visual Studio 2022 & CMake

Use Visual Studio's CMake to configure and build the project in **Release** mode:

```powershell
# 1. Configure the project
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 17 2022" -A x64 -DKIMODO_BUILD_TESTS=ON

# 2. Build all executables and libraries in Release mode
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release
```

*(Note: If your Visual Studio is installed on a different drive, e.g. `E:\Visual Studio\Product`, adjust the path to `cmake.exe` accordingly).*

This will generate:
- `build\Release\kimodo.dll` & `kimodo.lib` (Core engine)
- `build\bin\Release\ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll` (Tensor backend)
- `build\Release\kmd-generate.exe` (Text-to-motion CLI)
- `build\Release\kmd-inspect.exe` (Model inspector)

---

## 4. Download Model Weights

Download the **SOMA motion model** and the **Llama-3 text encoder bundle** using the Python downloader script:

```powershell
# Downloads SOMA RP v1.1 motion model and Llama-3 text encoder (~8 GB total)
python scripts/download_gguf_weights.py --model soma-rp-v1.1
```

Files will be placed in:
- `models/kimodo-soma-rp-v1.1-f32.gguf` (Motion checkpoint)
- `generated/llm2vec-text-bundle/` (Text encoder layers, tokenizer, embedding)

---

## 5. Generate Animations from a Text Prompt

Whenever you want to generate a new animation:

### Step A: Set Environment PATH
Add the compiled DLL folders to your active PowerShell session:
```powershell
$env:PATH = "E:\Kimodo\build\bin\Release;E:\Kimodo\build\Release;" + $env:PATH
```

### Step B: Create a Prompt
```powershell
"a person walking forward enthusiastically and waving their right hand" | Out-File -Encoding utf8 prompt.txt
```

### Step C: Run Motion Generation
```powershell
.\build\Release\kmd-generate.exe models\kimodo-soma-rp-v1.1-f32.gguf generated\llm2vec-text-bundle prompt.txt 120 50 42 output_motion\
```
> **Parameters:**
> - `120`: Frame count (~4 seconds at 30 FPS).
> - `50`: Diffusion steps (default: 50).
> - `42`: Random seed.
> - `output_motion\`: Destination folder for raw `.f32` motion streams.

---

## 6. Export to Unreal-Ready `.glb`

Convert the raw motion output into a compliant skinned glTF (`.glb`) file with correct rest-pose pelvis height:

```powershell
python scripts/export_glb.py --motion-dir output_motion --output output_motion/animation.glb
```

This creates **`E:\Kimodo\output_motion\animation.glb`**.

---

## 7. Import & Retarget in Unreal Engine 5

### Step A: Create or Open a UE5 Project
1. Open Unreal Engine 5.
2. Select **Games** -> **Third Person** -> **Blueprint**.
3. Create the project.

### Step B: Import the Animation
1. In the Content Browser, create a folder (e.g. `Content/KimodoAnimations/`).
2. **Drag and drop `E:\Kimodo\output_motion\animation.glb`** into the folder.
3. In the Import Dialog:
   - **Skeletal Mesh**: `Enabled (Checked)`
   - **Import Animations**: `Enabled (Checked)`
   - **Skeleton**: Leave as `None` (Unreal will auto-create the SOMA skeleton on first import).
   - Click **Import All**.

### Step C: Retarget to the Mannequin (`SKM_Manny` / `SKM_Quinn`)
1. Right-click the newly imported **`animation_Anim`** -> **Retarget Animations**.
2. In the Retarget Animations window:
   - Set **Target Skeletal Mesh**: select **`SKM_Manny_Simple`** or **`SKM_Manny`**.
   - In the asset list at the bottom right, **click `animation_Anim`** to highlight it.
   - Click the **Export Animations** button in the bottom right corner.
   - Choose your destination folder and save.

---

## 8. Play the Animation in Unreal Engine

### Quickest Test:
1. In your Content Browser, search for **`SKM_Manny`** and drag him into your level viewport.
2. Click on Manny in the level, and in the **Details** panel on the right:
   - Set **Animation Mode**: `Use Animation Asset`.
   - Set **Anim to Play**: Select your exported retargeted animation.
3. Press **Play (Alt + P)** to see Manny perform the animation live in your level!

### Triggering with a Key Press (e.g. Press 'E' to Play Montage):
1. Right-click your retargeted `AnimSequence` -> **Create** -> **Create AnimMontage**.
2. Open `Content/ThirdPerson/Blueprints/BP_ThirdPersonCharacter`.
3. In the Event Graph, add a **Keyboard E** event node -> drag off to **Play Anim Montage** -> select your Montage.
4. Press **Play**, move around with WASD, and press **E** to play your animation anytime!
