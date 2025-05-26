respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=2
optim_lr=0.01
#optim_lr=0.1
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

weight_jerk=0.1
weight_collision=0.1
weight_contact=0.1
weight_skate=0.0
contact_thresh=0.00
init_noise_scale=0.1
weight_interaction=1.0 
                    
# Path to interaction config - 确保使用我们更新的配置
interaction_cfg='/home/jixian22/Desktop/SCENE1/FOODLAB/high_five.json'

# Path to model checkpoint
model='./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'

# Create output directory if it doesn't exist
output_dir="./results/high_five_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir"

echo "Starting two-person high-five interaction optimization..."
echo "Output will be saved to: $output_dir"
echo "Using interaction config: $interaction_cfg"
echo "Optimization steps: $optim_steps"
echo "Learning rate: $optim_lr"
echo "Interaction weight: $weight_interaction"

# Run the high-five interaction optimization
python -m mld.optim_scene_mld \
    --denoiser_checkpoint "$model" \
    --interaction_cfg "$interaction_cfg" \
    --optim_lr $optim_lr \
    --optim_steps $optim_steps \
    --batch_size $batch_size \
    --guidance_param $guidance \
    --respacing "$respacing" \
    --export_smpl $export_smpl \
    --use_predicted_joints $use_predicted_joints \
    --optim_unit_grad $optim_unit_grad \
    --optim_anneal_lr $optim_anneal_lr \
    --weight_jerk $weight_jerk \
    --weight_collision $weight_collision \
    --weight_contact $weight_contact \
    --weight_interaction $weight_interaction \
    --contact_thresh $contact_thresh \
    --load_cache $load_cache \
    --init_noise_scale $init_noise_scale \
    --seed $seed

echo "Optimization completed!"
echo "Results saved to: $output_dir"

# Optional: Convert results to visualization format
if [ $export_smpl -eq 1 ]; then
    echo "SMPL files have been exported for visualization"
    echo "You can use Blender or other tools to visualize the results"
fi
