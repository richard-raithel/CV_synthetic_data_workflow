import os
from PIL import Image
import argparse


def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths


def align_images(a_file_paths, b_file_paths, target_path, image_size):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):
        img_a = Image.open(a_file_paths[i])
        img_b = Image.open(b_file_paths[i])
        if img_a.size == img_b.size and img_a.size == (image_size, image_size):
            print('Source Images are already: ', (image_size, image_size))
        else:
            # ADD CODE IN HERE TO ADJUST SQUARE & SQUEEZE
            img_a = img_a.resize((image_size, image_size))
            img_b = img_b.resize((image_size, image_size))

        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        help='Which folder to process (it should have subfolders testA, testB, trainA and trainB'
    )
    parser.add_argument('--resize', dest='target_image_size', help='Sets image size to align with MRCNN Model Input')
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    image_size = int(args.target_image_size)

    test_a_path = os.path.join(dataset_folder, 'testA')
    test_b_path = os.path.join(dataset_folder, 'testB')
    test_a_file_paths = get_file_paths(test_a_path)
    test_b_file_paths = get_file_paths(test_b_path)
    assert(len(test_a_file_paths) == len(test_b_file_paths))
    test_path = os.path.join(dataset_folder, 'test')

    train_a_path = os.path.join(dataset_folder, 'trainA')
    train_b_path = os.path.join(dataset_folder, 'trainB')
    train_a_file_paths = get_file_paths(train_a_path)
    train_b_file_paths = get_file_paths(train_b_path)
    assert(len(train_a_file_paths) == len(train_b_file_paths))
    train_path = os.path.join(dataset_folder, 'train')

    align_images(test_a_file_paths, test_b_file_paths, test_path, image_size)
    align_images(train_a_file_paths, train_b_file_paths, train_path, image_size)