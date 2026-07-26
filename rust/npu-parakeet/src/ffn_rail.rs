use std::{error::Error, fmt};

use ndarray::Array2;

const K768_D_MODEL: usize = 768;
const K768_D_FF: usize = 3072;
const K768_BLOCK: usize = 32;
const K768_FC1_K: usize = 800;
const TILE_M: usize = 64;
const AIE_ROWS: usize = 4;
const PAD_M_MULTIPLE: usize = TILE_M * AIE_ROWS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnActivation {
    Gelu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulMode {
    Identity,
    Gelu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatmulArtifact {
    pub stem: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub tiles: [usize; 3],
    pub cores: usize,
    pub mode: MatmulMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfnRailArtifacts {
    pub fc1: MatmulArtifact,
    pub cast_dff: String,
    pub fc2: MatmulArtifact,
    pub residual_add: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FfnRailConfig {
    pub pad_m: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub k_block: usize,
    pub activation: FfnActivation,
    pub residual_scale: f32,
    pub artifacts: FfnRailArtifacts,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FfnRailError {
    InvalidPadM {
        pad_m: usize,
        multiple: usize,
    },
    InvalidDimension {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidArtifact {
        artifact: &'static str,
        expected: String,
        actual: String,
    },
    InvalidConfig {
        field: &'static str,
        expected: &'static str,
    },
    ShapeMismatch {
        tensor: &'static str,
        expected_rows: Option<usize>,
        expected_columns: usize,
        actual_rows: usize,
        actual_columns: usize,
    },
    TooManyRows {
        tensor: &'static str,
        rows: usize,
        pad_m: usize,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FfnRailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPadM { pad_m, multiple } => {
                write!(f, "pad_m must be a nonzero multiple of {multiple}, got {pad_m}")
            }
            Self::InvalidDimension {
                field,
                expected,
                actual,
            } => write!(f, "{field} must be {expected}, got {actual}"),
            Self::InvalidArtifact {
                artifact,
                expected,
                actual,
            } => write!(
                f,
                "{artifact} artifact stem must be {expected}, got {actual}"
            ),
            Self::InvalidConfig { field, expected } => {
                write!(f, "{field} must be {expected}")
            }
            Self::ShapeMismatch {
                tensor,
                expected_rows,
                expected_columns,
                actual_rows,
                actual_columns,
            } => match expected_rows {
                Some(expected_rows) => write!(
                    f,
                    "{tensor} must have shape [{expected_rows}, {expected_columns}], got [{actual_rows}, {actual_columns}]"
                ),
                None => write!(
                    f,
                    "{tensor} must have {expected_columns} columns, got shape [{actual_rows}, {actual_columns}]"
                ),
            },
            Self::TooManyRows {
                tensor,
                rows,
                pad_m,
            } => write!(
                f,
                "{tensor} has {rows} rows, exceeding configured pad_m {pad_m}"
            ),
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(f, "{tensor} must have length {expected}, got {actual}"),
        }
    }
}

impl Error for FfnRailError {}

impl FfnRailConfig {
    pub fn k768_gelu(pad_m: usize) -> Result<Self, FfnRailError> {
        let config = Self {
            pad_m,
            d_model: K768_D_MODEL,
            d_ff: K768_D_FF,
            k_block: K768_BLOCK,
            activation: FfnActivation::Gelu,
            residual_scale: 1.0,
            artifacts: FfnRailArtifacts {
                fc1: MatmulArtifact {
                    stem: format!("{pad_m}x800x3072_64x32x128_8c_modalgelu"),
                    m: pad_m,
                    k: K768_FC1_K,
                    n: K768_D_FF,
                    tiles: [64, 32, 128],
                    cores: 8,
                    mode: MatmulMode::Gelu,
                },
                cast_dff: format!("cast_{pad_m}x3072"),
                fc2: MatmulArtifact {
                    stem: format!("{pad_m}x3072x768_64x32x96_8c_modalid"),
                    m: pad_m,
                    k: K768_D_FF,
                    n: K768_D_MODEL,
                    tiles: [64, 32, 96],
                    cores: 8,
                    mode: MatmulMode::Identity,
                },
                residual_add: format!("resadd_{pad_m}x768_s100"),
            },
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FfnRailError> {
        if self.pad_m == 0 || !self.pad_m.is_multiple_of(PAD_M_MULTIPLE) {
            return Err(FfnRailError::InvalidPadM {
                pad_m: self.pad_m,
                multiple: PAD_M_MULTIPLE,
            });
        }

        validate_dimension("d_model", K768_D_MODEL, self.d_model)?;
        validate_dimension("d_ff", K768_D_FF, self.d_ff)?;
        validate_dimension("k_block", K768_BLOCK, self.k_block)?;
        if self.activation != FfnActivation::Gelu {
            return Err(FfnRailError::InvalidConfig {
                field: "activation",
                expected: "GELU",
            });
        }
        if self.residual_scale.to_bits() != 1.0f32.to_bits() {
            return Err(FfnRailError::InvalidConfig {
                field: "residual_scale",
                expected: "exactly 1.0",
            });
        }

        validate_dimension("fc1.m", self.pad_m, self.artifacts.fc1.m)?;
        validate_dimension("fc1.k", K768_FC1_K, self.artifacts.fc1.k)?;
        validate_dimension("fc1.n", K768_D_FF, self.artifacts.fc1.n)?;
        validate_matmul_contract("fc1", &self.artifacts.fc1, [64, 32, 128], MatmulMode::Gelu)?;
        validate_artifact(
            "fc1",
            format!("{}x800x3072_64x32x128_8c_modalgelu", self.pad_m),
            &self.artifacts.fc1.stem,
        )?;

        validate_dimension("fc2.m", self.pad_m, self.artifacts.fc2.m)?;
        validate_dimension("fc2.k", K768_D_FF, self.artifacts.fc2.k)?;
        validate_dimension("fc2.n", K768_D_MODEL, self.artifacts.fc2.n)?;
        validate_matmul_contract(
            "fc2",
            &self.artifacts.fc2,
            [64, 32, 96],
            MatmulMode::Identity,
        )?;
        validate_artifact(
            "fc2",
            format!("{}x3072x768_64x32x96_8c_modalid", self.pad_m),
            &self.artifacts.fc2.stem,
        )?;
        validate_artifact(
            "cast_dff",
            format!("cast_{}x3072", self.pad_m),
            &self.artifacts.cast_dff,
        )?;
        validate_artifact(
            "residual_add",
            format!("resadd_{}x768_s100", self.pad_m),
            &self.artifacts.residual_add,
        )?;
        Ok(())
    }
}

fn validate_dimension(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), FfnRailError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FfnRailError::InvalidDimension {
            field,
            expected,
            actual,
        })
    }
}

fn validate_artifact(
    artifact: &'static str,
    expected: String,
    actual: &str,
) -> Result<(), FfnRailError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FfnRailError::InvalidArtifact {
            artifact,
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn validate_matmul_contract(
    artifact: &'static str,
    descriptor: &MatmulArtifact,
    tiles: [usize; 3],
    mode: MatmulMode,
) -> Result<(), FfnRailError> {
    for (index, (&expected, &actual)) in tiles.iter().zip(descriptor.tiles.iter()).enumerate() {
        let field = match (artifact, index) {
            ("fc1", 0) => "fc1.tiles[0]",
            ("fc1", 1) => "fc1.tiles[1]",
            ("fc1", _) => "fc1.tiles[2]",
            ("fc2", 0) => "fc2.tiles[0]",
            ("fc2", 1) => "fc2.tiles[1]",
            ("fc2", _) => "fc2.tiles[2]",
            _ => "matmul.tiles",
        };
        validate_dimension(field, expected, actual)?;
    }
    validate_dimension(
        match artifact {
            "fc1" => "fc1.cores",
            "fc2" => "fc2.cores",
            _ => "matmul.cores",
        },
        8,
        descriptor.cores,
    )?;
    if descriptor.mode != mode {
        return Err(FfnRailError::InvalidConfig {
            field: match artifact {
                "fc1" => "fc1.mode",
                "fc2" => "fc2.mode",
                _ => "matmul.mode",
            },
            expected: match mode {
                MatmulMode::Identity => "identity",
                MatmulMode::Gelu => "GELU",
            },
        });
    }
    Ok(())
}

pub fn pack_fc1_input(
    config: &FfnRailConfig,
    x: &Array2<f32>,
) -> Result<Array2<f32>, FfnRailError> {
    config.validate()?;
    let (rows, columns) = x.dim();
    if columns != config.d_model {
        return Err(FfnRailError::ShapeMismatch {
            tensor: "x",
            expected_rows: None,
            expected_columns: config.d_model,
            actual_rows: rows,
            actual_columns: columns,
        });
    }
    if rows > config.pad_m {
        return Err(FfnRailError::TooManyRows {
            tensor: "x",
            rows,
            pad_m: config.pad_m,
        });
    }

    let mut packed = Array2::<f32>::zeros((config.pad_m, config.artifacts.fc1.k));
    for ((row, column), &value) in x.indexed_iter() {
        packed[(row, column)] = value;
    }
    for row in 0..rows {
        packed[(row, config.d_model)] = 1.0;
    }
    Ok(packed)
}

pub fn augment_fc1_weight(
    config: &FfnRailConfig,
    w1: &Array2<f32>,
    b1: &[f32],
) -> Result<Array2<f32>, FfnRailError> {
    config.validate()?;
    let (rows, columns) = w1.dim();
    if rows != config.d_model || columns != config.d_ff {
        return Err(FfnRailError::ShapeMismatch {
            tensor: "w1",
            expected_rows: Some(config.d_model),
            expected_columns: config.d_ff,
            actual_rows: rows,
            actual_columns: columns,
        });
    }
    if b1.len() != config.d_ff {
        return Err(FfnRailError::LengthMismatch {
            tensor: "b1",
            expected: config.d_ff,
            actual: b1.len(),
        });
    }

    let mut augmented = Array2::<f32>::zeros((config.artifacts.fc1.k, config.artifacts.fc1.n));
    for ((row, column), &value) in w1.indexed_iter() {
        augmented[(row, column)] = value;
    }
    for (column, &value) in b1.iter().enumerate() {
        augmented[(config.d_model, column)] = value;
    }
    Ok(augmented)
}

pub fn pack_residual_ingress(
    config: &FfnRailConfig,
    x: &Array2<f32>,
    b2: &[f32],
) -> Result<Array2<f32>, FfnRailError> {
    config.validate()?;
    let (rows, columns) = x.dim();
    if columns != config.d_model {
        return Err(FfnRailError::ShapeMismatch {
            tensor: "x",
            expected_rows: None,
            expected_columns: config.d_model,
            actual_rows: rows,
            actual_columns: columns,
        });
    }
    if rows > config.pad_m {
        return Err(FfnRailError::TooManyRows {
            tensor: "x",
            rows,
            pad_m: config.pad_m,
        });
    }
    if b2.len() != config.d_model {
        return Err(FfnRailError::LengthMismatch {
            tensor: "b2",
            expected: config.d_model,
            actual: b2.len(),
        });
    }

    let mut packed = Array2::<f32>::zeros((config.pad_m, config.d_model));
    for ((row, column), &value) in x.indexed_iter() {
        packed[(row, column)] = value + b2[column];
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Axis};

    use super::{
        augment_fc1_weight, pack_fc1_input, pack_residual_ingress, FfnActivation, FfnRailConfig,
        FfnRailError, MatmulMode,
    };

    #[test]
    fn k768_gelu_config_has_exact_physical_contract() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");

        assert_eq!(cfg.pad_m, 512);
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.d_ff, 3072);
        assert_eq!(cfg.k_block, 32);
        assert_eq!(cfg.activation, FfnActivation::Gelu);
        assert_eq!(cfg.residual_scale, 1.0);

        assert_eq!(cfg.artifacts.fc1.m, 512);
        assert_eq!(cfg.artifacts.fc1.k, 800);
        assert_eq!(cfg.artifacts.fc1.n, 3072);
        assert_eq!(cfg.artifacts.fc1.tiles, [64, 32, 128]);
        assert_eq!(cfg.artifacts.fc1.cores, 8);
        assert_eq!(cfg.artifacts.fc1.mode, MatmulMode::Gelu);
        assert_eq!(
            cfg.artifacts.fc1.stem,
            "512x800x3072_64x32x128_8c_modalgelu"
        );

        assert_eq!(cfg.artifacts.fc2.m, 512);
        assert_eq!(cfg.artifacts.fc2.k, 3072);
        assert_eq!(cfg.artifacts.fc2.n, 768);
        assert_eq!(cfg.artifacts.fc2.tiles, [64, 32, 96]);
        assert_eq!(cfg.artifacts.fc2.cores, 8);
        assert_eq!(cfg.artifacts.fc2.mode, MatmulMode::Identity);
        assert_eq!(cfg.artifacts.fc2.stem, "512x3072x768_64x32x96_8c_modalid");
        assert_eq!(cfg.artifacts.cast_dff, "cast_512x3072");
        assert_eq!(cfg.artifacts.residual_add, "resadd_512x768_s100");
        assert_eq!(cfg.validate(), Ok(()));
    }

    #[test]
    fn k768_gelu_config_rejects_invalid_padding() {
        assert_eq!(
            FfnRailConfig::k768_gelu(0),
            Err(FfnRailError::InvalidPadM {
                pad_m: 0,
                multiple: 256,
            })
        );
        assert_eq!(
            FfnRailConfig::k768_gelu(64),
            Err(FfnRailError::InvalidPadM {
                pad_m: 64,
                multiple: 256,
            })
        );
        assert!(FfnRailConfig::k768_gelu(1536).is_ok());
    }

    #[test]
    fn k768_gelu_config_validation_rejects_a_changed_physical_contract() {
        let mut cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        cfg.artifacts.fc1.k = 768;

        assert_eq!(
            cfg.validate(),
            Err(FfnRailError::InvalidDimension {
                field: "fc1.k",
                expected: 800,
                actual: 768,
            })
        );
    }

    #[test]
    fn pack_fc1_input_appends_one_and_zero_fills_the_physical_layout() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let x = Array2::from_shape_fn((2, 768), |(row, col)| (row * 1000 + col) as f32);

        let packed = pack_fc1_input(&cfg, &x).expect("valid fc1 input");

        assert_eq!(packed.dim(), (512, 800));
        for row in 0..2 {
            assert_eq!(
                packed.slice(ndarray::s![row, 0..768]),
                x.slice(ndarray::s![row, ..])
            );
            assert_eq!(packed[(row, 768)], 1.0);
            assert!(packed
                .slice(ndarray::s![row, 769..800])
                .iter()
                .all(|&value| value == 0.0));
        }
        assert!(packed
            .slice(ndarray::s![2.., ..])
            .axis_iter(Axis(0))
            .all(|row| row.iter().all(|&value| value == 0.0)));
    }

    #[test]
    fn pack_fc1_input_rejects_wrong_model_width() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let x = Array2::<f32>::zeros((2, 767));

        assert_eq!(
            pack_fc1_input(&cfg, &x),
            Err(FfnRailError::ShapeMismatch {
                tensor: "x",
                expected_rows: None,
                expected_columns: 768,
                actual_rows: 2,
                actual_columns: 767,
            })
        );
    }

    #[test]
    fn pack_fc1_input_rejects_rows_beyond_padding() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let x = Array2::<f32>::zeros((513, 768));

        assert_eq!(
            pack_fc1_input(&cfg, &x),
            Err(FfnRailError::TooManyRows {
                tensor: "x",
                rows: 513,
                pad_m: 512,
            })
        );
    }

    #[test]
    fn augment_fc1_weight_appends_bias_row_and_zero_fills_tail() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let w1 =
            Array2::from_shape_fn((768, 3072), |(row, col)| ((row * 3072 + col) % 4096) as f32);
        let b1: Vec<f32> = (0..3072).map(|index| -(index as f32)).collect();

        let augmented = augment_fc1_weight(&cfg, &w1, &b1).expect("valid fc1 weight");

        assert_eq!(augmented.dim(), (800, 3072));
        assert_eq!(augmented.slice(ndarray::s![0..768, ..]), w1.view());
        assert!(augmented
            .row(768)
            .iter()
            .zip(b1.iter())
            .all(|(&actual, &expected)| actual == expected));
        assert!(augmented
            .slice(ndarray::s![769.., ..])
            .iter()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn augment_fc1_weight_rejects_wrong_matrix_shape() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let w1 = Array2::<f32>::zeros((767, 3072));
        let b1 = vec![0.0; 3072];

        assert_eq!(
            augment_fc1_weight(&cfg, &w1, &b1),
            Err(FfnRailError::ShapeMismatch {
                tensor: "w1",
                expected_rows: Some(768),
                expected_columns: 3072,
                actual_rows: 767,
                actual_columns: 3072,
            })
        );
    }

    #[test]
    fn augment_fc1_weight_rejects_wrong_bias_length() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let w1 = Array2::<f32>::zeros((768, 3072));
        let b1 = vec![0.0; 3071];

        assert_eq!(
            augment_fc1_weight(&cfg, &w1, &b1),
            Err(FfnRailError::LengthMismatch {
                tensor: "b1",
                expected: 3072,
                actual: 3071,
            })
        );
    }

    #[test]
    fn pack_residual_ingress_adds_bias_for_real_rows_and_zero_fills_padding() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let x = Array2::from_shape_fn((2, 768), |(row, col)| (row * 768 + col) as f32);
        let b2: Vec<f32> = (0..768).map(|column| 0.25 - column as f32).collect();

        let packed = pack_residual_ingress(&cfg, &x, &b2).expect("valid residual ingress");
        let without_bias =
            pack_residual_ingress(&cfg, &x, &[0.0; 768]).expect("valid zero-bias residual ingress");

        assert_eq!(packed.dim(), (512, 768));
        for row in 0..2 {
            for column in 0..768 {
                assert_eq!(packed[(row, column)], x[(row, column)] + b2[column]);
            }
        }
        assert_ne!(
            packed.slice(ndarray::s![0..2, ..]),
            without_bias.slice(ndarray::s![0..2, ..])
        );
        assert!(packed
            .slice(ndarray::s![2.., ..])
            .iter()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn pack_residual_ingress_rejects_shape_mismatch() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let wrong_x = Array2::<f32>::zeros((3, 767));
        assert_eq!(
            pack_residual_ingress(&cfg, &wrong_x, &[0.0; 768]),
            Err(FfnRailError::ShapeMismatch {
                tensor: "x",
                expected_rows: None,
                expected_columns: 768,
                actual_rows: 3,
                actual_columns: 767,
            })
        );

        let x = Array2::<f32>::zeros((3, 768));
        assert_eq!(
            pack_residual_ingress(&cfg, &x, &[0.0; 767]),
            Err(FfnRailError::LengthMismatch {
                tensor: "b2",
                expected: 768,
                actual: 767,
            })
        );
    }

    #[test]
    fn pack_residual_ingress_rejects_rows_beyond_padding() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let x = Array2::<f32>::zeros((513, 768));

        assert_eq!(
            pack_residual_ingress(&cfg, &x, &[0.0; 768]),
            Err(FfnRailError::TooManyRows {
                tensor: "x",
                rows: 513,
                pad_m: 512,
            })
        );
    }

    #[test]
    fn residual_preassociation_has_negligible_bounded_f32_delta() {
        let cfg = FfnRailConfig::k768_gelu(512).expect("valid K=768 rail config");
        let rows = 17;
        let x = Array2::from_shape_fn((rows, 768), |(row, column)| {
            (((row * 37 + column * 13) % 257) as f32 - 128.0) / 128.0
        });
        let y = Array2::from_shape_fn((rows, 768), |(row, column)| {
            (((row * 19 + column * 29 + 7) % 251) as f32 - 125.0) / 96.0
        });
        let b2: Vec<f32> = (0..768)
            .map(|column| (((column * 11 + 3) % 127) as f32 - 63.0) / 256.0)
            .collect();
        let residual =
            pack_residual_ingress(&cfg, &x, &b2).expect("valid deterministic residual ingress");

        let mut squared_delta = 0.0f64;
        let mut squared_reference = 0.0f64;
        for row in 0..rows {
            for column in 0..768 {
                let preassociated = residual[(row, column)] + y[(row, column)];
                let host_order = x[(row, column)] + (y[(row, column)] + b2[column]);
                assert!(preassociated.is_finite());
                assert!(host_order.is_finite());
                squared_delta += f64::from(preassociated - host_order).powi(2);
                squared_reference += f64::from(host_order).powi(2);
            }
        }
        let rel_l2 = (squared_delta / squared_reference).sqrt();
        eprintln!("bounded residual association rel-L2: {rel_l2:.9e}");
        assert!(rel_l2 < 1.0e-6, "association rel-L2 {rel_l2} is too large");
    }
}
