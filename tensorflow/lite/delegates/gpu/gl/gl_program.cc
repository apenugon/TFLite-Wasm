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

#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"

#include <string>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

absl::Status CreateNewProgramId(GLuint* program_id) {
  //RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glCreateProgram, program_id));
  *program_id = glCreateProgram();
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (!*program_id) {
    return absl::UnknownError("Can't create opengl program: 0 program_id");
  }
  return absl::OkStatus();
}

absl::Status CheckProgramLinked(GLuint program_id) {
  GLint linked;
  glGetProgramiv(program_id, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
    return absl::OkStatus();
  }
  GLint info_size;
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_size);
  std::string errors;
  errors.resize(info_size + 1 /* plus \0 */);
  glGetProgramInfoLog(program_id, info_size + 1, nullptr, &errors[0]);
  // TODO(akulik): use glValidateProgram to gather more info.
  return absl::UnavailableError("Program is not properly linked: " + errors);
}

struct ParameterSetter {
  absl::Status operator()(int value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform1i, program_id, uniform_id,
    //                          value);
    glUseProgram(program_id);
    glUniform1i(uniform_id, value);
    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const int2& value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform2i, program_id, uniform_id,
    //                          value.x, value.y);
    //
    glUseProgram(program_id);
    glUniform2i(uniform_id, value.x, value.y);
    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const int4& value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform4i, program_id, uniform_id,
    //                          value.x, value.y, value.z, value.w);
    glUseProgram(program_id);
    glUniform4i(uniform_id, value.x, value.y, value.z, value.w);
    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const std::vector<int2>& value) {
    std::vector<GLint> ints(value.size() * 2, 0);
    for (int i = 0; i < value.size(); ++i) {
      ints[i * 2] = value[i].x;
      ints[i * 2 + 1] = value[i].y;
    }
    //return TFLITE_GPU_CALL_GL(glProgramUniform2iv, program_id, uniform_id,
    //                          ints.size(), ints.data());
    glUseProgram(program_id);
    glUniform2iv(uniform_id, ints.size(), ints.data());
    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(unsigned int value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform1ui, program_id, uniform_id,
    //                          value);
    glUseProgram(program_id);
    glUniform1ui(uniform_id, value);

    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const uint4& value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform4ui, program_id, uniform_id,
    //                          value.x, value.y, value.z, value.w);
    glUseProgram(program_id);
    glUniform4ui(uniform_id, value.x, value.y, value.z, value.w);

    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(float value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform1f, program_id, uniform_id,
    //                          value);
    glUseProgram(program_id);
    glUniform1f(uniform_id, value);

    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const float2& value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform2f, program_id, uniform_id,
    //                          value.x, value.y);
    glUseProgram(program_id);
    glUniform2f(uniform_id, value.x, value.y);
 
    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus(); 
  }

  absl::Status operator()(const float4& value) {
    //return TFLITE_GPU_CALL_GL(glProgramUniform4f, program_id, uniform_id,
    //                          value.x, value.y, value.z, value.w);
    glUseProgram(program_id);
    glUniform4f(uniform_id, value.x, value.y, value.z, value.w);

    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  absl::Status operator()(const std::vector<float4>& value) {
    std::vector<GLfloat> floats(value.size() * 4, 0);
    for (int i = 0; i < value.size(); ++i) {
      floats[i * 4] = value[i].x;
      floats[i * 4 + 1] = value[i].y;
      floats[i * 4 + 2] = value[i].z;
      floats[i * 4 + 3] = value[i].w;
    }
    //return TFLITE_GPU_CALL_GL(glProgramUniform4fv, program_id, uniform_id,
    //                          floats.size(), floats.data());
    glUseProgram(program_id);
    glUniform4fv(uniform_id, floats.size(), floats.data());

    RETURN_IF_ERROR(GetOpenGlErrors());
    return absl::OkStatus();
  }

  const GLuint program_id;
  const GLint uniform_id;
};

}  // namespace

absl::Status GlProgram::CreateWithShader(GlShader shader,
                                         GlProgram* gl_program) {
  GLuint program_id;
  RETURN_IF_ERROR(CreateNewProgramId(&program_id));

  // program_id needs to be properly deleted if there will be an error, hense
  // wrap program_id into Program.
  //GlProgram program(program_id);

  //RETURN_IF_ERROR(
  //    TFLITE_GPU_CALL_GL(glAttachShader, program.id(), shader.id()));
  //RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glLinkProgram, program.id()));
  glAttachShader(program_id, shader.id());
  RETURN_IF_ERROR(GetOpenGlErrors());

  glLinkProgram(program_id);
  RETURN_IF_ERROR(GetOpenGlErrors());
  
  auto status = CheckProgramLinked(program_id);
  if (status != absl::OkStatus()) {
    return status;
  }
  RETURN_IF_ERROR(GetOpenGlErrors());

  glValidateProgram(program_id);
 
  //*gl_program = std::move(program);
  //*gl_program = program;
  gl_program->setId(program_id);

  glValidateProgram(program_id);

  RETURN_IF_ERROR(GetOpenGlErrors());
  return absl::OkStatus();
}

absl::Status GlProgram::CreateWithBinaryShader(const BinaryShader& shader,
                                               GlProgram* gl_program) {
  GLuint program_id;
  RETURN_IF_ERROR(CreateNewProgramId(&program_id));

  // program_id needs to be properly deleted if there will be an error, hense
  // wrap program_id into Program.
  GlProgram program(program_id);

  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glProgramBinary, program.id(),
                                     shader.format(), shader.binary().data(),
                                     shader.binary().size()));
  RETURN_IF_ERROR(CheckProgramLinked(program.id()));

  *gl_program = std::move(program);
  return absl::OkStatus();
}

absl::Status GlProgram::GetBinary(BinaryShader* binary_shader) {
  GLint size = 0;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_GL(glGetProgramiv, id_, GL_PROGRAM_BINARY_LENGTH, &size));
  if (!size) {
    return absl::InternalError("Getting binary size failed.");
  }
  // TODO(akulik): call
  // glProgramParameteri(id_, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE)
  // before linking a program to increase chances of retrieving a binary.
  std::vector<uint8_t> binary(size);
  GLsizei returned_size;
  GLenum format;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetProgramBinary, id_, size,
                                     &returned_size, &format,
                                     reinterpret_cast<void*>(&binary[0])));
  if (size != returned_size) {
    return absl::InternalError("Getting binary is failed.");
  }
  *binary_shader = BinaryShader(format, std::move(binary));
  return absl::OkStatus();
}
/*
GlProgram::GlProgram(GlProgram&& program) : id_(program.id_) {
  program.id_ = 0;
}*/

void GlProgram::Invalidate() {
  if (id_) {
    glDeleteProgram(id_);
    id_ = 0;
  }
}
/*
GlProgram& GlProgram::operator=(GlProgram&& program) {
  if (this != &program) {
    Invalidate();
    std::swap(id_, program.id_);
  }
  return *this;
}*/

GlProgram::~GlProgram() { Invalidate(); }

absl::Status GlProgram::SetParameter(const Variable& param) {
  GLint uniform_location;
  //RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetUniformLocation, &uniform_location,
  //                                   id_, param.name.c_str()));
  uniform_location = glGetUniformLocation(id_, param.name.c_str());
  RETURN_IF_ERROR(GetOpenGlErrors());
  
  return absl::visit(ParameterSetter{id_, uniform_location}, param.value);
}

absl::Status GlProgram::Dispatch(const uint3& workgroups) const {
  if (workgroups.x == 0 || workgroups.y == 0 || workgroups.z == 0) {
    return absl::InvalidArgumentError("Invalid workgroups");
  }
  //RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glUseProgram, id_));
  //return TFLITE_GPU_CALL_GL(glDispatchCompute, workgroups.x, workgroups.y,
  //                          workgroups.z);
  glUseProgram(id_);
  RETURN_IF_ERROR(GetOpenGlErrors());

  GLint validated;
  glValidateProgram(id_);
  glGetProgramiv(id_, GL_VALIDATE_STATUS, &validated);
  if (validated == GL_FALSE)
  {
    return absl::UnavailableError("Not validated");
  }

  glDispatchCompute(workgroups.x, workgroups.y, workgroups.z);
  RETURN_IF_ERROR(GetOpenGlErrors());

  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
