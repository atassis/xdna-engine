// rust/npu-weights/src/onnx.rs
//
// Minimal hand-written prost message set for decoding ONNX model files to read their
// weight INITIALIZERS only (NOT for inference). Field numbers + wire types mirror the
// official onnx.proto3 schema (https://github.com/onnx/onnx, onnx/onnx.proto3). We only
// model the messages/fields the bake needs: ModelProto.graph (7), GraphProto.initializer
// (5), TensorProto (dims/data_type/raw_data/*_data/name/data_location/external_data), and
// the StringStringEntryProto used for external_data {location,offset,length}.
//
// Anything we do not need (node, value_info, attributes, segments, ...) is simply omitted;
// prost skips unknown fields on decode, so a partial schema decodes a full ONNX file fine.
//
// TensorProto.data_type values we handle (subset of onnx DataType): FLOAT=1, FLOAT16=10,
// BFLOAT16=16, INT64=7. data_location: DEFAULT=0 (inline), EXTERNAL=1 (sidecar file).
use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    // many scalar fields (ir_version=1, producer_name=2, ...) precede graph=7; all optional/skipped.
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    // node=1 (repeated NodeProto): decoded so an arch can map ANONYMOUS weight initializers
    // (e.g. `onnx::MatMul_6400`) back to a logical name via the node that consumes them. NeMo
    // FastConformer / GigaAM exports anonymise all MatMul (and some Conv) weights, so the only
    // way to name them is the node-path convention the Python oracle uses.
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, optional, tag = "3")]
    pub name: Option<String>,
    #[prost(string, optional, tag = "4")]
    pub op_type: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, optional, tag = "1")]
    pub key: Option<String>,
    #[prost(string, optional, tag = "2")]
    pub value: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, optional, tag = "2")]
    pub data_type: Option<i32>,
    // float_data=4 (packed), int64_data=7 (packed): the typed-field fallback when raw_data is empty.
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, optional, tag = "8")]
    pub name: Option<String>,
    #[prost(bytes = "vec", optional, tag = "9")]
    pub raw_data: Option<Vec<u8>>,
    // external_data=13 (repeated StringStringEntryProto), data_location=14 (enum DataLocation).
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(int32, optional, tag = "14")]
    pub data_location: Option<i32>,
}

// onnx TensorProto.DataType subset.
pub const DT_FLOAT: i32 = 1;
pub const DT_FLOAT16: i32 = 10;
pub const DT_BFLOAT16: i32 = 16;
pub const DT_INT64: i32 = 7;
// onnx TensorProto.DataLocation.
pub const LOC_EXTERNAL: i32 = 1;

/// Decode a ModelProto from the .onnx protobuf bytes (the small graph file; external data stays in
/// its sidecar and is resolved separately).
pub fn decode_model(bytes: &[u8]) -> anyhow::Result<ModelProto> {
    Ok(ModelProto::decode(bytes)?)
}
