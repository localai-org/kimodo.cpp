// The byte-level BPE merge priority and Llama-3 pre-tokenization ordering
// were independently implemented with llama.cpp src/llama-vocab.cpp at
// 78ec4c378031811671d1c76a067acbee4f4c56ce as a reference.  No llama.cpp
// source is copied here.
#include "llm_tokenizer.hpp"

#include <gguf.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace kimodo::detail {
namespace {
constexpr char separator = '\x1f';

std::string utf8(std::uint32_t codepoint) {
    std::string result;
    if (codepoint < 0x80) result.push_back(static_cast<char>(codepoint));
    else if (codepoint < 0x800) { result.push_back(static_cast<char>(0xc0 | (codepoint >> 6))); result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f))); }
    else { result.push_back(static_cast<char>(0xe0 | (codepoint >> 12))); result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f))); result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f))); }
    return result;
}
bool ascii_alpha(unsigned char c) { return std::isalpha(c) != 0 || c >= 0x80; }
bool ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool ascii_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
bool contraction(std::string_view text, size_t at, size_t &length) {
    if (at >= text.size() || text[at] != '\'') return false;
    const auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    const std::string_view rest = text.substr(at);
    for (const std::string_view option : {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"}) {
        if (rest.size() >= option.size() && std::equal(option.begin(), option.end(), rest.begin(), [&](char a, char b) { return a == lower(b); })) { length = option.size(); return true; }
    }
    return false;
}
bool valid_utf8(std::string_view text) {
    for (size_t i = 0; i < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[i]);
        if (first < 0x80) { ++i; continue; }
        unsigned continuation = 0;
        if (first >= 0xc2 && first <= 0xdf) continuation = 1;
        else if (first >= 0xe0 && first <= 0xef) continuation = 2;
        else if (first >= 0xf0 && first <= 0xf4) continuation = 3;
        else return false;
        if (i + continuation >= text.size()) return false;
        for (unsigned offset = 1; offset <= continuation; ++offset)
            if ((static_cast<unsigned char>(text[i + offset]) & 0xc0) != 0x80) return false;
        // Reject overlong encodings, surrogate scalars, and code points above
        // U+10FFFF.  Byte-BPE itself operates on bytes, but the public API's
        // contract is UTF-8 and must reject malformed caller input.
        if ((first == 0xe0 && static_cast<unsigned char>(text[i + 1]) < 0xa0) ||
            (first == 0xed && static_cast<unsigned char>(text[i + 1]) >= 0xa0) ||
            (first == 0xf0 && static_cast<unsigned char>(text[i + 1]) < 0x90) ||
            (first == 0xf4 && static_cast<unsigned char>(text[i + 1]) >= 0x90)) return false;
        i += continuation + 1;
    }
    return true;
}
}

struct llm_tokenizer::impl {
    std::unordered_map<std::string, int> token_ids;
    std::unordered_map<std::string, int> merge_ranks;
    std::array<std::string, 256> byte_encode;
    int bos = 128000;
};

llm_tokenizer::~llm_tokenizer() = default;

std::expected<std::unique_ptr<llm_tokenizer>, std::string> llm_tokenizer::load(std::string_view path) {
    gguf_init_params params{false, nullptr};
    gguf_context *file = gguf_init_from_file(std::string(path).c_str(), params);
    if (!file) return std::unexpected("cannot load tokenizer GGUF");
    const auto release = std::unique_ptr<gguf_context, decltype(&gguf_free)>(file, gguf_free);
    auto key = [&](const char *name) { const auto value = gguf_find_key(file, name); if (value < 0) throw std::runtime_error(std::string("missing tokenizer key: ") + name); return value; };
    try {
        const auto architecture = key("general.architecture");
        if (std::string_view(gguf_get_val_str(file, architecture)) != "kimodo-llm2vec-tokenizer") return std::unexpected("not a Kimodo LLM2Vec tokenizer GGUF");
        const auto tokens_key = key("kimodo.tokenizer.tokens"), merges_key = key("kimodo.tokenizer.merges");
        if (gguf_get_arr_type(file, tokens_key) != GGUF_TYPE_STRING || gguf_get_arr_type(file, merges_key) != GGUF_TYPE_STRING || gguf_get_arr_n(file, tokens_key) != 128000 || gguf_get_arr_n(file, merges_key) != 280147) return std::unexpected("invalid Llama-3 tokenizer GGUF arrays");
        auto result = std::unique_ptr<llm_tokenizer>(new llm_tokenizer);
        result->impl_ = std::make_unique<impl>();
        result->impl_->token_ids.reserve(128000);
        result->impl_->merge_ranks.reserve(280147);
        for (size_t i = 0; i < 128000; ++i) {
            const char *value = gguf_get_arr_str(file, tokens_key, i);
            if (!value || !result->impl_->token_ids.emplace(value, static_cast<int>(i)).second) return std::unexpected("invalid duplicate tokenizer token");
        }
        for (size_t i = 0; i < 280147; ++i) {
            std::string merge = gguf_get_arr_str(file, merges_key, i);
            const auto split = merge.find(' ');
            if (split == std::string::npos || split == 0 || split + 1 == merge.size()) return std::unexpected("invalid BPE merge");
            merge[split] = separator;
            if (!result->impl_->merge_ranks.emplace(std::move(merge), static_cast<int>(i)).second) return std::unexpected("duplicate BPE merge");
        }
        std::array<bool, 256> direct{};
        for (unsigned i = 33; i <= 126; ++i) direct[i] = true;
        for (unsigned i = 161; i <= 172; ++i) direct[i] = true;
        for (unsigned i = 174; i <= 255; ++i) direct[i] = true;
        std::uint32_t extra = 256;
        for (unsigned i = 0; i < 256; ++i) result->impl_->byte_encode[i] = utf8(direct[i] ? i : extra++);
        return result;
    } catch (const std::exception &error) { return std::unexpected(error.what()); }
}

std::expected<std::vector<int>, std::string> llm_tokenizer::encode(std::string_view text) const {
    if (!impl_) return std::unexpected("invalid tokenizer");
    if (!valid_utf8(text)) return std::unexpected("prompt is not valid UTF-8");
    std::vector<std::string> words;
    for (size_t pos = 0; pos < text.size();) {
        size_t size = 0;
        if (contraction(text, pos, size)) { words.emplace_back(text.substr(pos, size)); pos += size; continue; }
        const unsigned char first = static_cast<unsigned char>(text[pos]);
        const bool prefixed_letter = !ascii_digit(first) && first != '\r' && first != '\n' && !ascii_alpha(first) && pos + 1 < text.size() && ascii_alpha(static_cast<unsigned char>(text[pos + 1]));
        if (ascii_alpha(first) || prefixed_letter) { size = prefixed_letter ? 1 : 0; while (pos + size < text.size() && ascii_alpha(static_cast<unsigned char>(text[pos + size]))) ++size; words.emplace_back(text.substr(pos, size)); pos += size; continue; }
        if (ascii_digit(first)) { while (size < 3 && pos + size < text.size() && ascii_digit(static_cast<unsigned char>(text[pos + size]))) ++size; words.emplace_back(text.substr(pos, size)); pos += size; continue; }
        if (!ascii_space(first) || (pos + 1 < text.size() && !ascii_space(static_cast<unsigned char>(text[pos + 1])) && !ascii_alpha(static_cast<unsigned char>(text[pos + 1])) && !ascii_digit(static_cast<unsigned char>(text[pos + 1])))) {
            size = first == ' ' ? 1 : 0; while (pos + size < text.size() && !ascii_space(static_cast<unsigned char>(text[pos + size])) && !ascii_alpha(static_cast<unsigned char>(text[pos + size])) && !ascii_digit(static_cast<unsigned char>(text[pos + size]))) ++size; words.emplace_back(text.substr(pos, size)); pos += size; continue;
        }
        while (pos + size < text.size() && ascii_space(static_cast<unsigned char>(text[pos + size]))) ++size;
        words.emplace_back(text.substr(pos, size)); pos += size;
    }
    std::vector<int> result{impl_->bos};
    for (const auto &word : words) {
        std::vector<std::string> symbols;
        for (unsigned char byte : word) symbols.push_back(impl_->byte_encode[byte]);
        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max(); size_t best = symbols.size();
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto found = impl_->merge_ranks.find(symbols[i] + separator + symbols[i + 1]);
                if (found != impl_->merge_ranks.end() && found->second < best_rank) { best_rank = found->second; best = i; }
            }
            if (best == symbols.size()) break;
            symbols[best] += symbols[best + 1]; symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
        }
        for (const auto &symbol : symbols) {
            const auto found = impl_->token_ids.find(symbol);
            if (found == impl_->token_ids.end()) return std::unexpected("BPE symbol is absent from vocabulary");
            result.push_back(found->second);
        }
    }
    return result;
}

} // namespace kimodo::detail
