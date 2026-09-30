# South-ece: evo evaluation against the supplied PPK reference

Evaluated the completed 2026-09-27 **f64 stereo-inertial run**, both raw VIO and final loop-corrected SLAM, against the supplied `ground-truth.csv`. **Translation ATE RMSE is 0.781 m raw and 0.787 m corrected**, with scale fixed to one.

These are results on the available PPK positions, not certification of full-recording accuracy. The supplied reference has missing intervals, contains antenna positions without a measured antenna-to-IMU offset, and none of its rows pass the exporter's additional `quality_accepted` checks.

## evo results

All numbers below are metres. Each trajectory receives one rigid SE(3) alignment over the **same 656 matched epochs**. No scale fitting, fitted time shift, outlier removal, per-segment alignment, or interpolation through reference gaps is used.

| Metric | Raw VIO | Loop-corrected SLAM |
|---|---:|---:|
| 3D ATE / APE RMSE | 0.781 | 0.787 |
| Mean | 0.684 | 0.687 |
| Median | 0.557 | 0.558 |
| 95th percentile | 1.408 | 1.428 |
| Maximum | 1.655 | 1.796 |
| Horizontal RMSE | 0.762 | 0.773 |
| Vertical RMSE | 0.170 | 0.148 |

The single accepted loop leaves aggregate ATE essentially unchanged: corrected RMSE is 0.006 m higher on this reference. Horizontal/vertical values are decompositions in the reference ENU frame after the same 3D alignment; they do not come from separate planar fits.

Both measured RMSE values are below the 4 m target **on the matched portion**. The unavailable final 65.5 seconds and reference-quality limitations prevent claiming the full sequence meets that target.

![evo report](/home/mikexyl/workspaces/visloc_ws/src/visloc-rs/results/ucy/south_ece_stereo_20260927/evo_ppk_20260927/evo_report.png)

## Reference, timing, and coverage

Reference supplied by the user:

`/data/ucy/South-ece-rtk-test/ppk/20260927-111745-152afc/ground-truth.csv`

The accompanying export identifies these as combined RTKLIB fixed solutions (`q=1`). All **659 supplied rows** were used as candidates, respecting the export's selection rule. Additional diagnostic policy flags were retained rather than silently filtering the data. All 659 have `quality_accepted=False`, and the strict-quality subset is empty. That flag is distinct from RTKLIB's fixed-solution status.

- 656 of 659 reference epochs matched; 3 had no estimate within 20 ms. Both evaluated trajectories use identical correspondences.
- Median absolute association difference: 10.208 ms; maximum: 16.662 ms. Estimates remain at their measured timestamps; positions are not resampled to force exact matches.
- Matched reference spans 0.193–174.193 s of the 239.716 s selected recording. There are internal gaps, the largest 31.4 s, and no supplied reference for the final 65.524 s. Fixed rows cover 55.0% of the original 1,198 GNSS epochs.
- The ground-truth `t` column is GNSS-derived UTC Unix time. Raw VIO timestamps were converted back from the replay clock using the saved integer offset; corrected trajectory timestamps already used the original clock. The calibrated camera/IMU offset was already represented and was not applied twice.
- No empirical time-offset optimization was performed. The recording's clock check used GPS UTC over USB without PPS; sample reception time is not substituted for measurement time.
- The six bag-file sizes and camera topic counts in the PPK audit match the completed replacement recording. Ground truth was used only for this offline evaluation, never fed into the estimator.

## Frames and metric

ECEF metre coordinates were translated to the first supplied ECEF point and rotated into local ENU. This is a rigid coordinate conversion; it does not alter scale or Euclidean errors.

The estimates describe the IMU/body origin, while PPK describes the GNSS antenna. No measured lever arm was supplied, so it remains uncompensated. A global alignment can absorb a constant displacement but does not remove the rotating antenna offset during turns. Results therefore include any contribution from that offset.

The reference provides no orientation. Its TUM identity quaternions are format placeholders only. **Only translation APE is evaluated**; no orientation or relative-pose accuracy is inferred from those placeholders.

Used **evo 1.37.1**, its `evo_ape` parser/implementation, and `evo_res` to generate standard result archives, metric plots and the comparison table. In [evo's documented alignment convention](https://github.com/MichaelGrupp/evo/wiki/Metrics), `--align` performs rigid SE(3) alignment; `--correct_scale` was not enabled. The scripts use headless plotting settings in memory.

## Artifacts and reproduction

Directory: `results/ucy/south_ece_stereo_20260927/evo_ppk_20260927/`.

- `evo_report.png` / `.pdf`: coverage-aware trajectory and error figure.
- `raw_vio_ape.zip`, `slam_corrected_ape.zip`: standard evo results, including associations, aligned trajectories and alignment matrices.
- `raw_vio_ape.pdf`, `slam_corrected_ape.pdf`, `evo_comparison.pdf`: native evo plots.
- `evo_results.csv`, `report.json`, `*_matched_errors.csv`: statistics and per-correspondence errors.
- `*.tum`, `reference/`, `input_audit.json`, `manifest.json`, `requirements.lock.txt`: exact evaluation inputs and provenance.
- `prepare.py`, `run_evo.py`, `report.py`, `commands.txt`: conversion, evo execution and analysis.

From that directory, the equivalent metric commands are:

```bash
evo_ape tum ppk_reference_enu.tum raw_vio.tum \
  --align --pose_relation trans_part --t_max_diff 0.02 --save_results raw_vio_ape.zip
evo_ape tum ppk_reference_enu.tum slam_corrected.tum \
  --align --pose_relation trans_part --t_max_diff 0.02 --save_results slam_corrected_ape.zip
evo_res raw_vio_ape.zip slam_corrected_ape.zip --use_filenames --save_table evo_results.csv
```

The run used the isolated `.runtime/evo-venv` environment. Saved evo error arrays were independently checked against the aligned position differences; RMSE recomputation agrees, both correspondence sets match exactly, and both alignment rotation matrices are proper rotations with unit scale.
