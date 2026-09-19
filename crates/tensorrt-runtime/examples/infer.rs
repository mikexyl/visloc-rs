use visloc_tensorrt::{Input, Session};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let plan = args
        .next()
        .ok_or("usage: infer model.plan input_name dim,dim,...")?;
    let name = args.next().ok_or("missing input name")?;
    let shape: Vec<i64> = args
        .next()
        .ok_or("missing dimensions")?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let bytes = visloc_tensorrt::tensor_bytes(&shape, visloc_tensorrt::DataType::F32)?;
    let mut session = Session::from_file(plan, 0)?;
    println!("Tensors: {:#?}", session.tensors());
    let data = vec![0.0f32; bytes / 4];
    for output in session.run(&[Input::f32(&name, &shape, &data)], 0)? {
        println!(
            "{}: {:?}, {:?}, {} bytes",
            output.info.name,
            output.info.dtype,
            output.info.shape,
            output.data.len()
        );
    }
    Ok(())
}
