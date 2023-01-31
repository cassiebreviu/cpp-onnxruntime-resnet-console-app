// OnnxRuntimeResNet.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <onnxruntime_cxx_api.h>
#include <iostream>

#include "Helpers.cpp"

int main()
{
	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;


    const std::string imageFile = "C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\dog.png";
    const std::string labelFile = "C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\imagenet_classes.txt";
    auto modelPath = L"C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\resnet50v2.onnx";


    //load labels
    std::vector<std::string> labels = loadLabels(labelFile);
    if (labels.empty()) {
        std::cout << "Failed to load labels: " << labelFile << std::endl;
        return 1;
    }

    // load image
    const std::vector<float> imageVec = loadImage(imageFile);
    if (imageVec.empty()) {
        std::cout << "Failed to load image: " << imageFile << std::endl;
        return 1;
    }

    if (imageVec.size() != numInputElements) {

        std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
        return 1;
    }

    // Use CUDA GPU
    Ort::SessionOptions ort_session_options;

    OrtCUDAProviderOptions options;
    options.device_id = 0;
    //options.arena_extend_strategy = 0;
    //options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
    //options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    //options.do_copy_in_default_stream = 1;
    
    OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);

    // create session
    session = Ort::Session(env, modelPath, ort_session_options);

    // Use CPU
    //session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

    // define shape
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // define array
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    std::copy(imageVec.begin(), imageVec.end(), input.begin());



    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get()};
    const std::array<const char*, 1> outputNames = { outputName.get()};
    inputName.release();
    outputName.release();


    // run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

     // sort results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // show Top5
    for (size_t i = 0; i < 5; ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
    }
}