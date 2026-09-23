import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def generate_spectrum_image(peaks, img_size=(224, 224), dpi=100):
    """生成干净的质谱图"""
    mz_values = [p['mz'] for p in peaks]
    intensities = [p['intensity'] for p in peaks]

    fig, ax = plt.subplots(figsize=(img_size[0] / dpi, img_size[1] / dpi), dpi=dpi)

    ax.vlines(mz_values, 0, intensities, colors='white', linewidth=1.5, alpha=0.9)
    ax.fill_between(mz_values, 0, intensities, color='white', alpha=0.2)

    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    mz_min, mz_max = min(mz_values), max(mz_values)
    mz_range = mz_max - mz_min
    if mz_range > 0:
        ax.set_xlim(mz_min - mz_range * 0.05, mz_max + mz_range * 0.05)

    intensity_max = max(intensities)
    if intensity_max > 0:
        ax.set_ylim(0, intensity_max * 1.05)

    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                pad_inches=0, facecolor='black', edgecolor='none')
    buf.seek(0)

    img = Image.open(buf)
    if img.size != img_size:
        img = img.resize(img_size, Image.Resampling.LANCZOS)

    img_array = np.array(img)
    plt.close(fig)

    return img_array


def generate_envelope_dict_vit(data, output_dir='envelope_plots_vit',
                               img_size=(224, 224), dpi=100,
                               save_images=True, save_base64=True,
                               return_arrays=True):
    """生成适合ViT输入的envelope字典"""
    envelopes = data.get('envelopes', [])
    output_path = Path(output_dir)

    if save_images:
        output_path.mkdir(exist_ok=True)

    envelope_dict = {}

    for idx, env in enumerate(envelopes):
        env_id = env['id']
        peaks = env['env_peaks']

        img_array = generate_spectrum_image(peaks, img_size, dpi)

        env_entry = {
            'info': {
                'id': env_id,
                'mono_mass': env.get('mono_mass', None),
                'charge': env.get('charge', None),
                'peak_count': len(peaks),
                'img_size': img_size
            }
        }

        if save_images:
            #img_path = output_path / f'envelope_{env_id:03d}.png'
            img_path = output_path / f'envelope_{int(env_id):03d}.png'
            Image.fromarray(img_array).save(img_path)
            env_entry['image_path'] = str(img_path.absolute())

        if save_base64:
            buffer = io.BytesIO()
            Image.fromarray(img_array).save(buffer, format='png')
            buffer.seek(0)
            env_entry['image_base64'] = base64.b64encode(buffer.read()).decode('utf-8')

        if return_arrays:
            env_entry['image_array'] = img_array

        envelope_dict[env_id] = env_entry

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(envelopes)} envelopes...")

    print(f"Done！")
    return envelope_dict