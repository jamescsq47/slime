# PD experiment scripts

- `baseline/`: launchers that use only the unmodified `pd_baseline` SGLang environment.
- `new_method/`: request-generation/direct/Host-pipeline launchers using the modified `pd` environment.
- `bandwidth/`: isolated GPU, Host and Mooncake transport microbenchmarks.
- `tools/`: analysis, validation and GPU-holder utilities.
- `common/`: process lifecycle helpers shared by the launchers.

The root contains only runtime modules imported by the workload.  Historical
compatibility entry points were removed; all experiments must use this tree.
