fn main() {
    let result = (|| -> visloc_ros::AnyResult<()> {
        let path = std::env::var("VISLOC_ROBOT_CONFIG")?;
        let config = serde_json::from_slice(&std::fs::read(path)?)?;
        visloc_ros::robot::run(config)
    })();
    if let Err(e) = result {
        eprintln!("visloc robot: {e}");
        std::process::exit(1);
    }
}
