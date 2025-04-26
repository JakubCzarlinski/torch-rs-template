use tch;
fn main() {
    // Check ENV variable for libtorch
    let libtorch_path = std::env::var("LIBTORCH");
    if libtorch_path.is_err() {
        println!("LIBTORCH environment variable not set");
        return;
    } else {
        println!(
            "LIBTORCH environment variable set to: {}",
            libtorch_path.unwrap()
        );
    }

    if tch::Cuda::is_available() {
        println!("CUDA is available");
    } else {
        println!("CUDA is not available");
    }

    let num_devices = tch::Cuda::device_count();
    println!("Number of CUDA devices: {}", num_devices);

    let dummy = &[1.0, 2.0, 3.0, 4.0];
    let example_tensor = tch::Tensor::from_slice(dummy);

    println!("Example tensor (DEFAULT): {:?}", example_tensor);

    let cpu_tensor = example_tensor.f_to_device(tch::Device::Cpu);
    if cpu_tensor.is_err() {
        println!("Error converting to CPU. I'm confused.");
    } else {
        println!("Example tensor (CPU): {:?}", cpu_tensor.unwrap());
    }
    let cuda_tensor = example_tensor.f_to_device(tch::Device::Cuda(0));
    if cuda_tensor.is_err() {
        println!("Error converting to CUDA.");
    } else {
        println!("Example tensor (CUDA): {:?}", cuda_tensor.unwrap());
    }
    let mps_tensor = example_tensor.f_to_device(tch::Device::Mps);
    if mps_tensor.is_err() {
        println!("Error converting to MPS.");
    } else {
        println!("Example tensor (MPS): {:?}", mps_tensor.unwrap());
    }
    let vulkan_tensor = example_tensor.f_to_device(tch::Device::Vulkan);
    if vulkan_tensor.is_err() {
        println!("Error converting to Vulkan.");
    } else {
        println!("Example tensor (Vulkan): {:?}", vulkan_tensor.unwrap());
    }

}
