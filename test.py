import fire

def scan_img(img_filename="test.jpg", threshold=0.5, stable_dif_print=False):
    from PIL import Image
    import numpy as np
    import torch
    import tqdm

    import deep_danbooru_model

    model = deep_danbooru_model.DeepDanbooruModel()
    model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

    model.eval()
    model.half()
    model.cuda()

    pic = Image.open(img_filename).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
    with torch.no_grad(), torch.autocast("cuda"):
        x = torch.from_numpy(a).cuda()
        # first run
        y = model(x)[0].detach().cpu().numpy()

        # measure performance
        for n in tqdm.tqdm(range(10)):
            model(x)

    output_array = [ (i, p) for i, p in enumerate(y) ]
    output_array.sort(key=lambda x: x[1], reverse=True)

    if stable_dif_print:
        print( ", ".join([ f"({model.tags[i]}:p)" for i, p in output_array if p >= threshold ]) )
    else:
        for i, p in output_array:
            if p >= threshold:
                print(model.tags[i], p)


if __name__ == '__main__':
    fire.Fire(scan_img)
