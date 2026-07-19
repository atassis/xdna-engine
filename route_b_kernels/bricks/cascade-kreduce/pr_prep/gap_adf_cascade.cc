// GAP: aie_api's only cascade surface is the ADF graph path, which needs <adf.h>
// (unvendored in this fork). A bare-metal/IRON kernel cannot reach cascade here.
#include <aie_api/aie_adf.hpp>   // -> adf/stream.hpp -> #include <adf.h>  (NOT FOUND)
extern "C" void uses_adf_cascade() {}
