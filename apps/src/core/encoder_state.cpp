#include "core/encoder_state.hpp"
#include "post/hf_adaptor.hpp"

namespace rmps {

void EncoderState::hf_adaptor_apply() {
    if (!hf_adaptor) hf_adaptor = std::make_unique<post::HFAdaptor>();
    hf_adaptor->apply(*this);
}

} // namespace rmps
