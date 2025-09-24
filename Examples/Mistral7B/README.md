### Export Mistral 7B Instruct v0.3

```shell
✗ python export.py

Loading checkpoint shards: 100%|███████████████████████████| 3/3 [00:12<00:00,  4.11s/it]
Converting PyTorch Frontend ==> MIL Ops: 100%|███| 5575/5575 [00:02<00:00, 2440.66 ops/s]
Running MIL frontend_pytorch pipeline: 100%|██████████| 5/5 [00:00<00:00,  7.12 passes/s]
Running MIL default pipeline: 100%|█████████████████| 79/79 [02:36<00:00,  1.98s/ passes]
Running MIL backend_mlprogram pipeline: 100%|███████| 12/12 [00:00<00:00, 22.90 passes/s]
Running compression: 100%|███████████████████████████| 296/296 [03:04<00:00,  1.60 ops/s]
...
```

### Generate Text

```shell
✗ swift run transformers "Best recommendations for a place to visit in Paris in August 2024:" --max-length 128 StatefulMistral7BInstructInt4.mlpackage

Best recommendations for a place to visit in Paris in August 2024:

1. Palace of Versailles: This iconic palace is a must-visit. It's a short train ride from Paris and offers a glimpse into the opulence of the French monarchy.

2. Eiffel Tower: No trip to Paris is complete without a visit to the Eiffel Tower. You can take an elevator ride to the top for a stunning view of the city.

3. Louvre Museum: Home to thousands of works of art, including the Mona Lisa and the Winged Victory of Samothrace, the Louvre is a cultural treasure.
```
