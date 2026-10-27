# Estado y Decisiones del Proyecto HMM 2D/3D

## Progreso por fases
- **Fase 0–1 (Especificación y Reglas)**: Completada (`docs/SPEC.pdf`, `.agent/rules/proyecto-hmm.md`).
- **Fase 2 (Configuración y FeatureExtractor 2D)**: Completada (`config/config.toml`, `core/config.py`, `core/FeatureExtractor.py`, `tests/test_feature_extractor.py`).
- **Fase 3 (Motor Matemático HMM)**: Completada (`core/HMMEngine.py`, `core/topology.py`, `tests/test_hmm_engine.py`).
- **Fase 4 (Captura, Aumento, Entrenamiento y Evaluación 2D)**: Completada (`training/augment.py`, `training/capture_tool.py`, `training/train.py`, `training/evaluate.py`, `models/2d_*.json`, `tests/test_training_pipeline.py`).
- **Fase 5 (Modo 3D sobre la esfera)**: Completada (`FeatureExtractor3D` y `ExtractionResult3D` en `core/FeatureExtractor.py`, mapeo 2D → esfera y aumento 3D en `training/augment.py`, entrenamiento `models/3d_*.json` en `training/train.py` y evaluación 3D en `training/evaluate.py`).
- **Fase 6 (GUI PySide6 + pyqtgraph.opengl)**: Completada (`main.py`, `gui/MainMenu.py`, `gui/View2D.py`, `gui/View3D.py`, `gui/RankingPanel.py`, `tests/test_gui.py`).
- **Fase 7 (Documentación de defensa en tablero)**: Completada (`docs/ecuaciones.md`).

## Estado general
Todas las fases del proyecto (Fases 0 a 7) se encuentran completadas y verificadas con pruebas automatizadas.
