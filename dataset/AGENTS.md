# Dataset workspace

Keep new collection scripts, dataset tools, tests, documentation, raw data and reports under this directory. Import the production B implementation at ../latest/openpi--main/openarm_cloth rather than copying the simulator, recognizer or controller here. The optional run_human_task.run(args, dataset_capture=...) hook is the integration boundary; default production launch behavior stays unchanged.

Keep raw sessions immutable except explicit, separate review annotations. Synthetic fixtures, pilot recordings and held-out evaluation references never become formal training data by changing labels. Use config/stages.json for distinct versioned stage names. Target-state smoothing after recording is not a valid physical demonstration until replayed and re-evaluated. Preserve failure logs and raw accepted targets. Store no generated dataset or personal calibration on a public remote without the user's explicit request.
