VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    TfLite*;
    *tflite*;

  # Hide everything else.
  local:
    *;
};
