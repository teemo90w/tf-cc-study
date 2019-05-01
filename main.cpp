#include <stdlib.h>
#include <iostream>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;

void
usage()
{
    printf("please set function name: session | const | variable | matrix | placeholder | example | sample\n");
    exit(0);
}

int
sample()
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }

    printf("[%s] jobs done.\n", __func__);
}

int
debug_session()
{
    auto root = Scope::NewRootScope();
    auto p_session = new ClientSession(root);
    delete p_session;

    printf("[%s] take over\n", __func__);
    return 0;
}

int
debug_const()
{
    auto root = Scope::NewRootScope();
    auto w = Const(root, 2, {});
    auto p_session = new ClientSession(root);
    vector<Tensor> outputs;
    p_session->Run({w}, &outputs);
    LOG(INFO) << "w = " << outputs[0].scalar<int>();
    delete p_session;

    return 0;
}

int
debug_variable()
{
    auto root = Scope::NewRootScope();
    auto x = Variable(root, {}, DataType::DT_INT32);
    auto assign_x = Assign(root, x, 3); // initializer for x
    auto y = Variable(root, {2, 3}, DataType::DT_FLOAT);
    auto assign_y = Assign(root, y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT)); // initializer for y
    auto p_session = new ClientSession(root);
    p_session->Run({assign_x, assign_y}, nullptr); // initialize
    vector<Tensor> outputs;
    p_session->Run({x, y}, &outputs);
    LOG(INFO) << "x = " << outputs[0].scalar<int>();
    LOG(INFO) << "y = " << outputs[1].matrix<float>();
    delete p_session;

    return 0;
}

int
debug_matrix()
{
    auto root = Scope::NewRootScope();
    auto x = Variable(root, {5, 2}, DataType::DT_FLOAT);
    auto assign_x = Assign(root, x, RandomNormal(root, {5, 2}, DataType::DT_FLOAT));
    auto y = Variable(root, {2, 3}, DataType::DT_FLOAT);
    auto assign_y = Assign(root, y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT));
    auto xy = MatMul(root, x, y);
    auto z = Const(root, 2.f, {5, 3});
    auto xyz = Add(root, xy, z);
    auto p_session = new ClientSession(root);
    p_session->Run({assign_x, assign_y}, nullptr);
    vector<Tensor> outputs;
    p_session->Run({x, y, z, xy, xyz}, &outputs);
    LOG(INFO) << "x = " << outputs[0].matrix<float>();
    LOG(INFO) << "y = " << outputs[1].matrix<float>();
    LOG(INFO) << "xy = " << outputs[3].matrix<float>();
    LOG(INFO) << "z = " << outputs[2].matrix<float>();
    LOG(INFO) << "xyz = " << outputs[4].matrix<float>();
    delete p_session;

    return 0;
}

int
debug_placeholder()
{
    auto root = Scope::NewRootScope();
    auto x = Placeholder(root, DataType::DT_INT32);
    auto w = Const(root, 1, {1, 2});
    auto wx = MatMul(root, x, w);
    auto b = Const(root, 2, {2});
    auto wx_b = Add(root, wx, b);
    auto p_session = new ClientSession(root);
    vector<Tensor> outputs;
    p_session->Run({{x, {{1}, {1}, {1}}}}, {wx, wx_b}, &outputs);
    LOG(INFO) << "wx = " << outputs[0].matrix<int>();
    LOG(INFO) << "wx_b = " << outputs[1].matrix<int>();
    delete p_session;

    return 0;
}

// 官方的例子
int
tf_example()
{
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();

  printf("[%s] take over\n", __func__);

  return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        usage();
    }

    const char *fn = argv[1];
    printf("fn: %s\n", fn);
    if (strcmp(fn, "session") == 0) {
        debug_session();
    } else if (strcmp(fn, "const") == 0) {
        debug_const();
    } else if (strcmp(fn, "variable") == 0) {
        debug_variable();
    } else if (strcmp(fn, "matrix") == 0) {
        debug_matrix();
    } else if (strcmp(fn, "placeholder") == 0) {
        debug_placeholder();
    } else if (strcmp(fn, "example") == 0) {
        tf_example();
    } else if (strcmp(fn, "sample") == 0) {
        sample();
    }else {
        usage();
    }

    return 0;
}
