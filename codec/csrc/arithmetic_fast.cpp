#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr uint32_t kTotal = 1u << 15;
constexpr uint64_t kHalf = 1ull << 31;
constexpr uint64_t kQuarter = 1ull << 30;
constexpr uint64_t kThreeQuarter = 3ull << 30;

uint32_t probability_to_frequency(float probability) {
    if (!(probability == probability)) {
        probability = 0.5f;
    }
    if (probability <= 0.0f) {
        return 1;
    }
    if (probability >= 1.0f) {
        return kTotal - 1;
    }
    const double value = static_cast<double>(probability);
    int64_t frequency = static_cast<int64_t>(value * static_cast<double>(kTotal) + 0.5);
    if (frequency <= 0) {
        return 1;
    }
    if (frequency >= static_cast<int64_t>(kTotal)) {
        return kTotal - 1;
    }
    return static_cast<uint32_t>(frequency);
}

class RangeEncoderFast {
public:
    void encode(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits,
        py::array_t<float, py::array::c_style | py::array::forcecast> probabilities) {
        const auto bit_buffer = bits.request();
        const auto probability_buffer = probabilities.request();
        if (bit_buffer.size != probability_buffer.size) {
            throw std::invalid_argument("bits and probabilities must have the same size");
        }
        const auto* bit_data = static_cast<const uint8_t*>(bit_buffer.ptr);
        const auto* probability_data = static_cast<const float*>(probability_buffer.ptr);
        for (py::ssize_t index = 0; index < bit_buffer.size; ++index) {
            encode_bit(bit_data[index] != 0, probability_data[index]);
        }
    }

    py::bytes finish() {
        if (finished_) {
            throw std::runtime_error("range encoder has already been finalized");
        }
        ++pending_bits_;
        flush_pending(low_ < kQuarter ? 0 : 1);
        if (bit_count_ != 0) {
            bit_buffer_ <<= (8 - bit_count_);
            output_.push_back(static_cast<uint8_t>(bit_buffer_ & 0xffu));
            bit_buffer_ = 0;
            bit_count_ = 0;
        }
        finished_ = true;
        return py::bytes(reinterpret_cast<const char*>(output_.data()), output_.size());
    }

private:
    void emit_bit(uint32_t bit) {
        bit_buffer_ = (bit_buffer_ << 1) | (bit & 1u);
        ++bit_count_;
        if (bit_count_ == 8) {
            output_.push_back(static_cast<uint8_t>(bit_buffer_ & 0xffu));
            bit_buffer_ = 0;
            bit_count_ = 0;
        }
    }

    void flush_pending(uint32_t bit) {
        emit_bit(bit);
        const uint32_t inverse = bit ^ 1u;
        while (pending_bits_ != 0) {
            emit_bit(inverse);
            --pending_bits_;
        }
    }

    void encode_bit(bool bit, float probability) {
        if (finished_) {
            throw std::runtime_error("cannot encode after finish");
        }
        const uint64_t frequency_one = probability_to_frequency(probability);
        const uint64_t frequency_zero = kTotal - frequency_one;
        const uint64_t lower_cumulative = bit ? frequency_zero : 0;
        const uint64_t upper_cumulative = bit ? kTotal : frequency_zero;
        const uint64_t range = high_ - low_ + 1;
        high_ = low_ + (range * upper_cumulative / kTotal) - 1;
        low_ = low_ + (range * lower_cumulative / kTotal);

        while (true) {
            if (high_ < kHalf) {
                flush_pending(0);
            } else if (low_ >= kHalf) {
                flush_pending(1);
                low_ -= kHalf;
                high_ -= kHalf;
            } else if (low_ >= kQuarter && high_ < kThreeQuarter) {
                ++pending_bits_;
                low_ -= kQuarter;
                high_ -= kQuarter;
            } else {
                break;
            }
            low_ = (low_ << 1) & 0xffffffffull;
            high_ = ((high_ << 1) | 1ull) & 0xffffffffull;
        }
    }

    uint64_t low_ = 0;
    uint64_t high_ = 0xffffffffull;
    uint64_t pending_bits_ = 0;
    std::vector<uint8_t> output_;
    uint32_t bit_buffer_ = 0;
    uint32_t bit_count_ = 0;
    bool finished_ = false;
};

class RangeDecoderFast {
public:
    explicit RangeDecoderFast(py::bytes data) : input_(static_cast<std::string>(data)) {
        for (int index = 0; index < 32; ++index) {
            code_ = ((code_ << 1) | read_bit()) & 0xffffffffull;
        }
    }

    py::array_t<uint8_t> decode(
        py::array_t<float, py::array::c_style | py::array::forcecast> probabilities) {
        const auto probability_buffer = probabilities.request();
        const auto* probability_data = static_cast<const float*>(probability_buffer.ptr);
        py::array_t<uint8_t> output(probability_buffer.size);
        auto output_buffer = output.request();
        auto* output_data = static_cast<uint8_t*>(output_buffer.ptr);
        for (py::ssize_t index = 0; index < probability_buffer.size; ++index) {
            output_data[index] = decode_bit(probability_data[index]);
        }
        return output;
    }

private:
    uint32_t read_bit() {
        if (bit_mask_ == 0) {
            current_byte_ = byte_position_ < input_.size()
                ? static_cast<uint8_t>(input_[byte_position_++])
                : 0;
            bit_mask_ = 0x80u;
        }
        const uint32_t bit = (current_byte_ & bit_mask_) != 0 ? 1u : 0u;
        bit_mask_ >>= 1;
        return bit;
    }

    uint8_t decode_bit(float probability) {
        const uint64_t frequency_one = probability_to_frequency(probability);
        const uint64_t frequency_zero = kTotal - frequency_one;
        const uint64_t range = high_ - low_ + 1;
        const uint64_t scaled = ((code_ - low_ + 1) * kTotal - 1) / range;
        const bool bit = scaled >= frequency_zero;
        const uint64_t lower_cumulative = bit ? frequency_zero : 0;
        const uint64_t upper_cumulative = bit ? kTotal : frequency_zero;
        high_ = low_ + (range * upper_cumulative / kTotal) - 1;
        low_ = low_ + (range * lower_cumulative / kTotal);

        while (true) {
            if (high_ < kHalf) {
                // No offset adjustment.
            } else if (low_ >= kHalf) {
                low_ -= kHalf;
                high_ -= kHalf;
                code_ -= kHalf;
            } else if (low_ >= kQuarter && high_ < kThreeQuarter) {
                low_ -= kQuarter;
                high_ -= kQuarter;
                code_ -= kQuarter;
            } else {
                break;
            }
            low_ = (low_ << 1) & 0xffffffffull;
            high_ = ((high_ << 1) | 1ull) & 0xffffffffull;
            code_ = ((code_ << 1) | read_bit()) & 0xffffffffull;
        }
        return bit ? 1u : 0u;
    }

    std::string input_;
    size_t byte_position_ = 0;
    uint32_t bit_mask_ = 0;
    uint8_t current_byte_ = 0;
    uint64_t low_ = 0;
    uint64_t high_ = 0xffffffffull;
    uint64_t code_ = 0;
};

}  // namespace

PYBIND11_MODULE(_arithmetic_fast, module) {
    module.attr("TOT_FREQ") = kTotal;
    py::class_<RangeEncoderFast>(module, "RangeEncoder")
        .def(py::init<>())
        .def("encode", &RangeEncoderFast::encode)
        .def("finish", &RangeEncoderFast::finish);
    py::class_<RangeDecoderFast>(module, "RangeDecoder")
        .def(py::init<py::bytes>())
        .def("decode", &RangeDecoderFast::decode);
}
