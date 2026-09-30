// rf_norm.cc at -Oz with every exported symbol prefixed rfz_, so one entry (the chain end's
// rf_ss_last, 2400 B at -O2, 464 at -Oz) can link beside the -O2 build on column 7.
#define rf_gain rfz_gain
#define rf_unit rfz_unit
#define rf_ss_first rfz_ss_first
#define rf_ss_mid rfz_ss_mid
#define rf_ss_last rfz_ss_last
#define rf_ss_first_w rfz_ss_first_w
#define rf_ss_mid_w rfz_ss_mid_w
#define rf_ss_last_w rfz_ss_last_w
#define rf_rstd_recv rfz_rstd_recv
#define rf_pre_scale rfz_pre_scale
#define rf_post_scale rfz_post_scale
#include "rf_norm.cc"
