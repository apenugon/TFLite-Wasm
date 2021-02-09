/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {

//GlShader::GlShader(GlShader&& shader) : id_(shader.id_) { shader.id_ = 0; }

void GlShader::Invalidate() {
  if (id_) {
    glDeleteShader(id_);
    id_ = 0;
  }
}
/*
GlShader& GlShader::operator=(GlShader&& shader) {
  if (this != &shader) {
    Invalidate();
    std::swap(id_, shader.id_);
  }
  return *this;
}
*/
GlShader::~GlShader() { Invalidate(); }

absl::Status GlShader::CompileShader(GLenum shader_type,
                                     const std::string& shader_source,
                                     GlShader* gl_shader) {
  // NOTE: code compilation can fail due to gl errors happened before
  GLuint shader_id;

  //RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glCreateShader, &shader_id, shader_type));
  shader_id = glCreateShader(shader_type);
  RETURN_IF_ERROR(GetOpenGlErrors());
  
  //GlShader shader(shader_id);

  const char* src = shader_source.c_str();
  //RETURN_IF_ERROR(
  //    TFLITE_GPU_CALL_GL(glShaderSource, shader.id(), 1, &src, nullptr));

  glShaderSource(shader_id, 1, &src, nullptr);
  RETURN_IF_ERROR(GetOpenGlErrors());

  glCompileShader(shader_id);

  // Didn't check for opengl errors here because we want to get better logs
  // if it didn't compile.
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    return absl::InternalError("Did not compile!");
    GLint info_log_len = 0;
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_len);
    std::string errors(info_log_len, 0);
    glGetShaderInfoLog(shader_id, info_log_len, nullptr, &errors[0]);
    return absl::InternalError("Shader compilation failed: " + errors +
                               "\nProblem shader is:\n" + shader_source);
  }



  //*gl_shader = std::move(shader);
  //gl_shader = shader;
  gl_shader->setId(shader_id);


  //glGetShaderiv(gl_shader->id(), GL_COMPILE_STATUS, &compiled);

  //RETURN_IF_ERROR(GetOpenGlErrors());
  //return absl::InternalError("Unclear error");

  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
