# Changelog

## 2026-03-11 20:20 IST
- rewrote `talkingdata_preprocessor.py` to expose the requested function-based API and chunked TalkingData loading
- rebuilt `02_comprehensive_preprocessing.py` as a CLI pipeline that appends real TalkingData human sessions before augmentation
- expanded `preprocessing_module.py` with schema alignment, dual-mode sequence preparation, device/network synthesis, validation reporting, and merge helpers
- generated fresh artifacts in `preprocessing_output/` including the unified CSV, sequence payload, and integration/validation reports
- added `tests/test_preprocessing.py` for sessionization, schema, and label-distribution regression coverage
