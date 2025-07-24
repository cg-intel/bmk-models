@echo off
mkdir results 2>nul

for %%m in (models\*.onnx) do (
   for %%p in (f16 f32) do (
       echo Testing %%~nm with %%p precision...
       benchmark_app -m %%m -infer_precision %%p -d GPU -hint latency > results\%%~nm_%%p_GPU_latency.txt
   )
)