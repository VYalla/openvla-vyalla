# TODO 4-3-25 

- the run inference script was created to run inference on a single image (after downloading the openvla model) 
- make a new repo to put the inference script and run it 
- need to use the libero dataset (it is located at /home/vishal/Documents/openvla_testv1/LIBERO/libero/datasets) 
    - (or find a way to get a sample image to test with  
- use windsurf to run the run_inference script with cmd `python run_inference.py --image_path <path_to_image> --instruction <instruction>`
- afterwards run the libero eval scripts to test performance on dataset 
- next steps => figure out how to intergate the 'framework' paper methods into these models (aka figure out the internal workings)


# TODO 4-7-25

- data from the libero dataset was saved to /home/vishal/Documents/openvla_testv1/LIBERO/vyalla_scripts/all_demos 
- use sample images from that, and use it to run_inference.py 
    - this will give understanding of what the vla-action model outputs 
- use the conformal prediction code/method from the "robots that ask for help" paper, and figure out where to insert it in the inference pipeline 
```
-> when running the inference, are you running a model or a pipeline?


Based on the code in the run_inference.py script, you are running a model rather than a pipeline. Specifically, you're using the OpenVLA model (openvla/openvla-7b by default) through the Hugging Face Transformers library.

The key components showing this is a model-based approach:

1) You're loading a specific model using AutoModelForVision2Seq.from_pretrained()
2) You're using a processor from the same model using AutoProcessor.from_pretrained()

3) The inference is performed directly with the model via the vla.predict_action() method

This is a single model approach rather than a pipeline that would typically involve multiple models or processing steps chained together. The script loads the model, processes the input image and instruction, and then directly generates the action prediction in a single forward pass.
```
so will have to look at other ways to inject the conformal prediction method into the inference pipeline 


# TODO 4-8-25

play around with the run_sim script, here is the cmd used : conda activate libero && python run_sim.py --model_name "openvla/openvla-7b" --instruction "pick up the cube" --device "cuda:0" 
- make it so it picks the object up sucessfully (maybe use a general prompt instead of action specfic)

then go to the next part to use the conformal prediction methods

# TODO 4-10-25

to run the inference locally with log text, use: 
```
python inference_runtime/run_local_inference.py --image_path /home/vishal/Documents/openvla_testv1/LIBERO/vyalla_scripts/all_demos/ketchup/episode_demo_44/frame_0002.png --instruction "pick up the ketchup bottle" --checkpoint_path /home/vishal/Documents/openvla_testv1/openvla-vyalla/checkpoints/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt
```

next steps: 
- study the conformal prediction methods
- ideneify what exactly the llm is outputing (use the log.txt that is outputed in the local run), and how to inject the cp method into it 