#!/usr/bin/env python3
import argparse
import json
import shutil
import warnings
import random
from math import sqrt
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


# ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# set base path
base_path = Path(__file__).parent.parent


# CODE FROM COCO_JSON_UTILS - USED TO GENERATE METADATA
# REQUIRES:
# 1) mask_definition path
# 2) dataset_info path


class InfoJsonUtils:
    """ Creates an info object to describe a COCO dataset
    """
    def create_coco_info(self, description, url, version, year, contributor, date_created):
        """ Creates the "info" portion of COCO json
        """
        info = dict()
        info['description'] = description
        info['url'] = url
        info['version'] = version
        info['year'] = year
        info['contributor'] = contributor
        info['date_created'] = date_created

        return info


class LicenseJsonUtils:
    """ Creates a license object to describe a COCO dataset
    """
    def create_coco_license(self, url, license_id, name):
        """ Creates the "licenses" portion of COCO json
        """
        lic = dict()
        lic['url'] = url
        lic['id'] = license_id
        lic['name'] = name

        return lic


class CategoryJsonUtils:
    """ Creates a category object to describe a COCO dataset
    """
    def create_coco_category(self, supercategory, category_id, name):
        category = dict()
        category['supercategory'] = supercategory
        category['id'] = category_id
        category['name'] = name

        return category


class ImageJsonUtils:
    """ Creates an image object to describe a COCO dataset
    """
    def create_coco_image(self, image_path, image_id, image_license):
        """ Creates the "image" portion of COCO json
        """
        # Open the image and get the size
        image_file = Image.open(image_path)
        width, height = image_file.size

        image = dict()
        image['license'] = image_license
        image['file_name'] = image_path.name
        image['width'] = width
        image['height'] = height
        image['id'] = image_id

        return image


class AnnotationJsonUtils:
    """ Creates an annotation object to describe a COCO dataset
    """
    def __init__(self):
        self.annotation_id_index = 0

    def create_coco_annotations(self, image_mask_path, image_id, category_ids):
        """ Takes a pixel-based RGB image mask and creates COCO annotations.
        Args:
            image_mask_path: a pathlib.Path to the image mask
            image_id: the integer image id
            category_ids: a dictionary of integer category ids keyed by RGB color (a tuple converted to a string)
                e.g. {'(255, 0, 0)': {'category': 'owl', 'super_category': 'bird'} }
        Returns:
            annotations: a list of COCO annotation dictionaries that can
            be converted to json. e.g.:
            {
                "segmentation": [[101.79,307.32,69.75,281.11,...,100.05,309.66]],
                "area": 51241.3617,
                "iscrowd": 0,
                "image_id": 284725,
                "bbox": [68.01,134.89,433.41,174.77],
                "category_id": 6,
                "id": 165690
            }
        """
        # Set class variables
        self.image_id = image_id
        self.category_ids = category_ids

        # Make sure keys in category_ids are strings
        for key in self.category_ids.keys():
            if type(key) is not str:
                raise TypeError('category_ids keys must be strings (e.g. "(0, 0, 255)")')
            break

        # Open and process image
        self.mask_image = Image.open(image_mask_path)
        self.mask_image = self.mask_image.convert('RGB')
        self.width, self.height = self.mask_image.size

        # Split up the multi-colored masks into multiple 0/1 bit masks
        self._isolate_masks()

        # Create annotations from the masks
        self._create_annotations()

        return self.annotations

    def _isolate_masks(self):
        # Breaks mask up into isolated masks based on color

        self.isolated_masks = dict()
        # for x in range(self.width):
        #     for y in range(self.height):
        #         pixel_rgb = self.mask_image.getpixel((x,y))
        #         pixel_rgb_str = str(pixel_rgb)

        #         # If the pixel is any color other than black, add it to a respective isolated image mask
        #         if not pixel_rgb == (0, 0, 0):
        #             if self.isolated_masks.get(pixel_rgb_str) is None:
        #                 # Isolated mask doesn't have its own image yet, create one
        #                 # with 1-bit pixels, default black. Make room for 1 pixel of
        #                 # padding on each edge to allow the contours algorithm to work
        #                 # when shapes bleed up to the edge
        #                 self.isolated_masks[pixel_rgb_str] = Image.new('1', (self.width + 2, self.height + 2))

        #             # Add the pixel to the mask image, shifting by 1 pixel to account for padding
        #             self.isolated_masks[pixel_rgb_str].putpixel((x + 1, y + 1), 1)

        # This is a much faster way to split masks using Numpy
        arr = np.array(self.mask_image, dtype=np.uint32)
        rgb32 = (arr[:,:,0] << 16) + (arr[:,:,1] << 8) + arr[:,:,2]
        unique_values = np.unique(rgb32)
        for u in unique_values:
            if u != 0:
                r = int((u & (255 << 16)) >> 16)
                g = int((u & (255 << 8)) >> 8)
                b = int(u & 255)
                self.isolated_masks[str((r, g, b))] = np.equal(rgb32, u)

    def _create_annotations(self):
        # Creates annotations for each isolated mask

        # Each image may have multiple annotations, so create an array
        self.annotations = []
        for key, mask in self.isolated_masks.items():
            annotation = dict()
            annotation['segmentation'] = []
            annotation['iscrowd'] = 0
            annotation['image_id'] = self.image_id
            if not self.category_ids.get(key):
                print(f'category color not found: {key}; check for missing category or antialiasing')
                continue
            annotation['category_id'] = self.category_ids[key]
            annotation['id'] = self._next_annotation_id()

            # Find contours in the isolated mask
            mask = np.asarray(mask, dtype=np.float32)
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')

            polygons = []
            for contour in contours:
                # Flip from (row, col) representation to (x, y)
                # and subtract the padding pixel
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Make a polygon and simplify it
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)

                if (poly.area > 16): # Ignore tiny polygons
                    if (poly.geom_type == 'MultiPolygon'):
                        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                        poly = poly.convex_hull

                    if (poly.geom_type == 'Polygon'): # Ignore if still not a Polygon (could be a line or point)
                        polygons.append(poly)
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        annotation['segmentation'].append(segmentation)

            if len(polygons) == 0:
                # This item doesn't have any visible polygons, ignore it
                # (This can happen if a randomly placed foreground is covered up
                #  by other foregrounds)
                continue

            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            x, y, max_x, max_y = multi_poly.bounds
            self.width = max_x - x
            self.height = max_y - y
            annotation['bbox'] = (x, y, self.width, self.height)
            annotation['area'] = multi_poly.area

            # Finally, add this annotation to the list
            self.annotations.append(annotation)

    def _next_annotation_id(self):
        # Gets the next annotation id
        # Note: This is not a unique id. It simply starts at 0 and increments each time it is called

        a_id = self.annotation_id_index
        self.annotation_id_index += 1
        return a_id


class CocoJsonCreator:
    def validate_and_process_args(self, args):
        """ Validates the arguments coming in from the command line and performs
            initial processing
        Args:
            args: ArgumentParser arguments
        """
        # Validate the mask definition file exists
        mask_definition_file = Path(args.mask_definition)
        if not (mask_definition_file.exists and mask_definition_file.is_file()):
            raise FileNotFoundError(f'mask definition file was not found: {mask_definition_file}')

        # Load the mask definition json
        with open(mask_definition_file) as json_file:
            self.mask_definitions = json.load(json_file)

        self.dataset_dir = mask_definition_file.parent

        # Validate the dataset info file exists
        dataset_info_file = Path(args.dataset_info)
        if not (dataset_info_file.exists() and dataset_info_file.is_file()):
            raise FileNotFoundError(f'dataset info file was not found: {dataset_info_file}')

        # Load the dataset info json
        with open(dataset_info_file) as json_file:
            self.dataset_info = json.load(json_file)

        assert 'info' in self.dataset_info, 'dataset_info JSON was missing "info"'
        assert 'license' in self.dataset_info, 'dataset_info JSON was missing "license"'

    def create_info(self):
        """ Creates the "info" piece of the COCO json
        """
        info_json = self.dataset_info['info']
        iju = InfoJsonUtils()
        return iju.create_coco_info(
            description = info_json['description'],
            version = info_json['version'],
            url = info_json['url'],
            year = info_json['year'],
            contributor = info_json['contributor'],
            date_created = info_json['date_created']
        )

    def create_licenses(self):
        """ Creates the "license" portion of the COCO json
        """
        license_json = self.dataset_info['license']
        lju = LicenseJsonUtils()
        lic = lju.create_coco_license(
            url = license_json['url'],
            license_id = license_json['id'],
            name = license_json['name']
        )
        return [lic]

    def create_categories(self):
        """ Creates the "categories" portion of the COCO json
        Returns:
            categories: category objects that become part of the final json
            category_ids_by_name: a lookup dictionary for category ids based
                on the name of the category
        """
        cju = CategoryJsonUtils()
        categories = []
        category_ids_by_name = dict()
        category_id = 1 # 0 is reserved for the background

        super_categories = self.mask_definitions['super_categories']
        for super_category, _categories in super_categories.items():
            for category_name in _categories:
                categories.append(cju.create_coco_category(super_category, category_id, category_name))
                category_ids_by_name[category_name] = category_id
                category_id += 1

        return categories, category_ids_by_name

    def create_images_and_annotations(self, category_ids_by_name):
        """ Creates the list of images (in json) and the annotations for each
            image for the "image" and "annotations" portions of the COCO json
        """
        iju = ImageJsonUtils()
        aju = AnnotationJsonUtils()

        image_objs = []
        annotation_objs = []
        image_license = self.dataset_info['license']['id']
        image_id = 0

        mask_count = len(self.mask_definitions['masks'])
        print(f'Processing {mask_count} mask definitions...')

        # For each mask definition, create image and annotations
        for file_name, mask_def in tqdm(self.mask_definitions['masks'].items()):
            # Create a coco image json item
            image_path = Path(self.dataset_dir) / file_name
            image_obj = iju.create_coco_image(
                image_path,
                image_id,
                image_license)
            image_objs.append(image_obj)

            mask_path = Path(self.dataset_dir) / mask_def['mask']

            # Create a dict of category ids keyed by rgb_color
            category_ids_by_rgb = dict()
            for rgb_color, category in mask_def['color_categories'].items():
                category_ids_by_rgb[rgb_color] = category_ids_by_name[category['category']]
            annotation_obj = aju.create_coco_annotations(mask_path, image_id, category_ids_by_rgb)
            annotation_objs += annotation_obj # Add the new annotations to the existing list
            image_id += 1

        return image_objs, annotation_objs

    def main(self, args):
        self.validate_and_process_args(args)

        info = self.create_info()
        licenses = self.create_licenses()
        categories, category_ids_by_name = self.create_categories()
        images, annotations = self.create_images_and_annotations(category_ids_by_name)

        master_obj = {
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        # Write the json to a file
        output_path = Path(self.dataset_dir) / 'coco_instances.json'
        with open(output_path, 'w+') as output_file:
            json.dump(master_obj, output_file)

        print(f'Annotations successfully written to file:\n{output_path}')


# CODE FROM IMAGE_COMPOSITION - USED TO GENERATE SYNTHETIC IMAGES W/MASKS
# REQUIRES:
# 1) input_dir (path to source data)
# 2) output_dir (path to synthetic dataset save location)
# 3) count (number of images)
# 4) width (width of synthetic image)
# 5) height (height of synthetic image)
# 6) output_type (file type of synthetic image - png or jpg - defaults jpg)
# 7) silent (doesn't prompt for input)

class MaskJsonUtils:
    """ Creates a JSON definition file for image masks.
    """

    def __init__(self, output_dir):
        """ Initializes the class.
        Args:
            output_dir: the directory where the definition file will be saved
        """
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """ Adds a new category to the set of the corresponding super_category
        Args:
            category: e.g. 'eagle'
            super_category: e.g. 'bird'
        Returns:
            True if successful, False if the category was already in the dictionary
        """
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True  # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Takes an image path, its corresponding mask path, and its color categories,
            and adds it to the appropriate dictionaries
        Args:
            image_path: the relative path to the image, e.g. './images/00000001.png'
            mask_path: the relative path to the mask image, e.g. './masks/00000001.png'
            color_categories: the legend of color categories, for this particular mask,
                represented as an rgb-color keyed dictionary of category names and their super categories.
                (the color category associations are not assumed to be consistent across images)
        Returns:
            True if successful, False if the image was already in the dictionary
        """
        if self.masks.get(image_path):
            return False  # image/mask is already in the dictionary

        # Create the mask definition
        mask = {
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True  # Addition was successful

    def get_masks(self):
        """ Gets all masks that have been added
        """
        return self.masks

    def get_super_categories(self):
        """ Gets the dictionary of super categories for each category in a JSON
            serializable format
        Returns:
            A dictionary of lists of categories keyed on super_category
        """
        serializable_super_cats = dict()
        for super_cat, categories_set in self.super_categories.items():
            # Sets are not json serializable, so convert to list
            serializable_super_cats[super_cat] = list(categories_set)
        return serializable_super_cats

    def write_masks_to_json(self):
        """ Writes all masks and color categories to the output file path as JSON
        """
        # Serialize the masks and super categories dictionaries
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories()
        masks_obj = {
            'masks': serializable_masks,
            'super_categories': serializable_super_cats
        }

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'mask_definitions.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(masks_obj))


class ImageComposition:
    """ Composes images together in random ways, applying transformations to the foreground to create a synthetic
        combined image.
    """

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8  # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 30
        self.mask_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(0, self.max_foregrounds)]

        assert len(self.mask_colors) >= self.max_foregrounds, 'length of mask_colors should be >= max_foregrounds'

    def _validate_and_process_args(self, args):
        # Validates input arguments and sets up class variables
        # Args:
        #     args: the ArgumentParser command line arguments

        self.silent = args.silent

        # Validate the count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate the width and height
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        # Validate and process the output type
        if args.output_type is None:
            self.output_type = '.jpg'  # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validate and process output and input directories
        self._validate_and_process_output_directory(args)
        self._validate_and_process_input_directory(args)

    def _validate_and_process_output_directory(self, args):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.masks_output_dir = self.output_dir / 'masks'

        # Create directories
        try:
            self.output_dir.mkdir(exist_ok=False)
            self.images_output_dir.mkdir(exist_ok=False)
            self.masks_output_dir.mkdir(exist_ok=False)
        except FileExistsError:
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(exist_ok=False)
            self.images_output_dir.mkdir(exist_ok=False)
            self.masks_output_dir.mkdir(exist_ok=False)

        if not self.silent:
            # Check for existing contents in the images directory
            for _ in self.images_output_dir.iterdir():
                # We found something, check if the user wants to overwrite files or quit
                should_continue = input('output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_input_directory(self, args):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir does not exist: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if 'foregrounds' in x.name:
                self.foregrounds_dir = x
            elif 'backgrounds' in x.name:
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        # Expected directory structure:
        # + foregrounds_dir
        #     + super_category_dir
        #         + category_dir
        #             + foreground_image.png

        self.foregrounds_dict = dict()

        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                warnings.warn(
                    f'file found in foregrounds directory (expected super-category directories), ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(
                        f'file found in super category directory (expected category directories), ignoring: {category_dir}')
                    continue

                # This is a category directory
                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        warnings.warn(f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue

                    # Valid foreground image, add to foregrounds_dict
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(
                    f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generating {self.count} images with masks...')

        mju = MaskJsonUtils(self.output_dir)

        # Create all images/masks (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a background
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(2, self.max_foregrounds)
            foregrounds = []
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_path = random.choice(self.foregrounds_dict[super_category][category])

                # Get the color
                mask_rgb_color = self.mask_colors[fg_i]

                foregrounds.append({
                    'super_category': super_category,
                    'category': category,
                    'foreground_path': foreground_path,
                    'mask_rgb_color': mask_rgb_color
                })

            # Compose foregrounds and background
            composite, mask = self._compose_images(foregrounds, background_path)

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}'  # e.g. 00000023.jpg

            # # # ONLY NEED THE MASK TO BE SAVED SINCE MASKS ARE USED IN GAN OUTPUTS
            # # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}'  # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename  # e.g. my_output_dir/images/00000023.jpg
            # composite = composite.convert('RGB')  # remove alpha
            # composite.save(composite_path)

            # Save the mask image to the masks sub-directory
            mask_filename = f'{save_filename}.png'  # masks are always png to avoid lossy compression
            mask_path = self.output_dir / 'masks' / mask_filename  # e.g. my_output_dir/masks/00000023.png
            mask.save(mask_path)

            color_categories = dict()
            for fg in foregrounds:
                # Add category and color info
                mju.add_category(fg['category'], fg['super_category'])
                color_categories[str(fg['mask_rgb_color'])] = \
                    {
                        'category': fg['category'],
                        'super_category': fg['super_category']
                    }

            # Add the mask to MaskJsonUtils
            mju.add_mask(
                composite_path.relative_to(self.output_dir).as_posix(),
                mask_path.relative_to(self.output_dir).as_posix(),
                color_categories
            )

        # Write masks to json
        mju.write_masks_to_json()

    def _compose_images(self, foregrounds, background_path):
        # Composes a foreground image and a background image and creates a segmentation mask
        # using the specified color. Validation should already be done by now.
        # Args:
        #     foregrounds: a list of dicts with format:
        #       [{
        #           'super_category':super_category,
        #           'category':category,
        #           'foreground_path':foreground_path,
        #           'mask_rgb_color':mask_rgb_color
        #       },...]
        #     background_path: the path to a valid background image
        # Returns:
        #     composite: the composed image
        #     mask: the mask image

        # Open background and convert to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # get background size
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width
        max_crop_y_pos = bg_height

        # check that foreground is smaller than background
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        # crop_x_pos = 0  # random.randint(0+int(bg_width/4), max_crop_x_pos)
        # crop_y_pos = 0  # random.randint(0+int(bg_height/4), max_crop_y_pos)
        # composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))
        src_image_size = background.size
        composite = background.resize((self.width, self.height))
        composite_mask = Image.new('RGB', composite.size, 0)

        position_list_x = list()
        position_list_y = list()
        # print('NEXT IMAGE')
        for fg in foregrounds:
            # get foreground path and perform transformations
            fg_path = fg['foreground_path']
            fg_image = self._transform_foreground(fg, fg_path)

            scale = tuple(dimA/dimB for dimA, dimB in zip(fg_image.size, src_image_size))

            x = scale[0]*self.width
            y = scale[1]*self.height

            if x < 3 or y < 3:
                x = fg_image.size[0]
                y = fg_image.size[1]

            conformed_size = (int(x), int(y))
            fg_image = fg_image.resize(conformed_size)

            try:
                # open_cv_image = numpy.array(fg_image)
                # from scipy import sparse
                # remove_index = list()
                # for i in open_cv_image:
                #     # m = sparse.csr_matrix(i)
                #     # if sparse.issparse(m) is True:
                #     #     # print('sparse found')
                #     #     remove_index.append(i)
                #     if numpy.all(i==0):
                #         remove_index.append(i)
                # c = np.setdiff1d(open_cv_image, remove_index)
                # import torchvision.transforms as T
                # transform = T.ToPILImage()
                # fg_image = transform(c)

                # calculate allowable foreground location square

                padding = 40
                min_x_position = int(composite.size[0] * 0.15 + padding/2)
                min_y_position = int(composite.size[1] * 0.15 + padding/2)
                max_x_position = int((composite.size[0] / 2 * sqrt(2)) - fg_image.size[0] - padding)
                max_y_position = int((composite.size[1] / 2 * sqrt(2)) - fg_image.size[1] - padding)
                assert max_x_position >= 0 and max_y_position >= 0, \
                    f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.width}x{self.height}), check your input parameters'
                # paste_position = (min_x_position + random.randint(0, max_x_position), min_y_position + random.randint(0, max_y_position))
                # # get paste location min and max values
                # min_x = paste_position[0] - 2
                # min_y = paste_position[1] - 2
                # max_x = paste_position[0] + fg_image.size[0] + 2
                # max_y = paste_position[1] + fg_image.size[1] + 2
                #
                # # check if random foreground x location is already taken, pass if true
                #
                # x_list = list(range(min_x, max_x+1))
                # y_list = list(range(min_y, max_y+1))
                #
                # check = any(item in position_list_x for item in x_list) and any(item in position_list_y for item in y_list)

                # check = False
                # iterator = 0

                # while check is True:
                #     if iterator < 500:
                paste_position = (min_x_position + random.randint(0, max_x_position),
                                  min_y_position + random.randint(0, max_y_position))

                # get paste location min and max values
                # min_x = paste_position[0]
                # min_y = paste_position[1]
                # max_x = paste_position[0] + fg_image.size[0]
                # max_y = paste_position[1] + fg_image.size[1]

                # x_list = list(range(min_x, max_x + 1))
                # y_list = list(range(min_y, max_y + 1))

                        # check = any(item in position_list_x for item in x_list) and any(
                        #     item in position_list_y for item in y_list)

                    #     iterator += 1
                    # else:
                    #     break

                # if check is False:
                    # if successful foreground placement, add to list of taken pixels
                # for i in range(min_x, max_x+1):
                #     position_list_x.append(i)
                # for j in range(min_y, max_y+1):
                #     position_list_y.append(j)

                # print(list(range(min_x, max_x+1)))
                # print('position list x', position_list_x)
                # print(list(range(min_y, max_y+1)))
                # print('position list y', position_list_y)

                # Create a new foreground image as large as the composite and paste it on top
                new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
                new_fg_image.paste(fg_image, paste_position)

                # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
                alpha_mask = fg_image.getchannel(3)
                new_alpha_mask = Image.new('L', composite.size, color=0)
                new_alpha_mask.paste(alpha_mask, paste_position)
                composite = Image.composite(new_fg_image, composite, new_alpha_mask)

                # Grab the alpha pixels above a specified threshold
                alpha_threshold = 200
                mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
                uint8_mask = np.uint8(mask_arr)  # This is composed of 1s and 0s

                # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
                mask_rgb_color = fg['mask_rgb_color']
                red_channel = uint8_mask * mask_rgb_color[0]
                green_channel = uint8_mask * mask_rgb_color[1]
                blue_channel = uint8_mask * mask_rgb_color[2]
                rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
                isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
                isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

                composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

            except ValueError:
                max_x_position = composite.size[0] - fg_image.size[0]
                max_y_position = composite.size[1] - fg_image.size[1]
                # max_x_position = composite.size[0]
                # max_y_position = composite.size[1]
                assert max_x_position >= 0 and max_y_position >= 0, \
                    f'foreground {fg_path} is too big ({fg_image.size[0]}x{fg_image.size[1]}) ' \
                    f'for the requested output size ({self.width}x{self.height}), check your input parameters'

                paste_position = (random.randint(10, max_x_position), random.randint(10, max_y_position))

                # Create a new foreground image as large as the composite and paste it on top
                new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
                new_fg_image.paste(fg_image, paste_position)

                # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
                alpha_mask = fg_image.getchannel(3)
                new_alpha_mask = Image.new('L', composite.size, color=0)
                new_alpha_mask.paste(alpha_mask, paste_position)
                composite = Image.composite(new_fg_image, composite, new_alpha_mask)

                # Grab the alpha pixels above a specified threshold
                alpha_threshold = 200
                mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
                uint8_mask = np.uint8(mask_arr)  # This is composed of 1s and 0s

                # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
                mask_rgb_color = fg['mask_rgb_color']
                red_channel = uint8_mask * mask_rgb_color[0]
                green_channel = uint8_mask * mask_rgb_color[1]
                blue_channel = uint8_mask * mask_rgb_color[2]
                rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
                isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
                isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

                composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

        return composite, composite_mask

    def _transform_foreground(self, fg, fg_path):
        # Open foreground and get the alpha channel
        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        # assert np.any(fg_alpha == 0), f'foreground needs to have some transparency: {str(fg_path)}'

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # Scale the foreground
        # scale = random.random() * .5 + 1  # Pick something between .5 and 1
        # new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        #
        # fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .5 + .75  # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Add any other transformations here...

        return fg_image

    def _create_info(self):
        # A convenience wizard for automatically creating dataset info
        # The user can always modify the resulting .json manually if needed

        if self.silent:
            # SETTING DEFAULT INFO FILE
            # print('Note - USING DEFAULTS: you can always modify the json manually if you need to update this.')
            info = dict()
            info['description'] = 'COCO Based Synthetic Image Dataset for Mask RCNN Model'
            info['url'] = 'SEE FUCHS or NYE LUBRICANTS'
            info['version'] = 'DEFAULT'
            info['contributor'] = 'Script modified by RJB'
            now = datetime.now()
            info['year'] = now.year
            info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

            image_license = dict()
            image_license['id'] = 0

            image_license['name'] = 'Fuchs Lubricants'
            image_license['url'] = 'search for Computer Vision'

            dataset_info = dict()
            dataset_info['info'] = info
            dataset_info['license'] = image_license

            # Write the JSON output file
            output_file_path = Path(self.output_dir) / 'dataset_info.json'
            with open(output_file_path, 'w+') as json_file:
                json_file.write(json.dumps(dataset_info))

            print('Successfully created synthetic images')
            return

        should_continue = input('Would you like to create dataset info json? (y/n) ').lower()
        if should_continue != 'y' and should_continue != 'yes':
            print('No problem. You can always create the json manually.')
            quit()

        # print('Note: you can always modify the json manually if you need to update this.')
        info = dict()
        info['description'] = input('Description: ')
        info['url'] = input('URL: ')
        info['version'] = input('Version: ')
        info['contributor'] = input('Contributor: ')
        now = datetime.now()
        info['year'] = now.year
        info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

        image_license = dict()
        image_license['id'] = 0

        should_add_license = input('Add an image license? (y/n) ').lower()
        if should_add_license != 'y' and should_add_license != 'yes':
            image_license['url'] = ''
            image_license['name'] = 'None'
        else:
            image_license['name'] = input('License name: ')
            image_license['url'] = input('License URL: ')

        dataset_info = dict()
        dataset_info['info'] = info
        dataset_info['license'] = image_license

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'dataset_info.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(dataset_info))

        print('Successfully created synthetic images')

    # Start here
    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()
        self._create_info()
        print('Image composition completed.')


def generate(output_dir, count, data, size):
    # GENERATE SYNTHETIC TRAIN DATASET
    synthetic_compositions = ImageComposition()
    synth_args = argparse.Namespace(
        input_dir=data,
        output_dir=output_dir,
        count=count,
        width=size,
        height=size,
        output_type='png',
        silent=True
    )
    synthetic_compositions.main(synth_args)

    # GENERATE SYNTHETIC TRAIN DATASET METADATA LABELS
    # coco_json_metadata_generator = CocoJsonCreator()
    # coco_args = argparse.Namespace(
    #     mask_definition=f'{output_dir}/mask_definitions.json',
    #     dataset_info=f'{output_dir}/dataset_info.json'
    # )
    #
    # coco_json_metadata_generator.main(coco_args)


if __name__ == '__main__':
    # set DATASET_SIZE to desired value and run
    # DATASET_SIZE = 100
    # DATASET_SIZE_TRAIN = int(DATASET_SIZE * .8)
    # DATASET_SIZE_VAL = int(DATASET_SIZE * .2)
    # SOURCE_DATA = os.path.join(base_path, "dataset", "bg-fg")
    # OUTPUT_DIR_TRAIN = os.path.join(base_path, "dataset", "train-val-test", f"train{DATASET_SIZE_TRAIN}")
    # OUTPUT_DIR_VAL = os.path.join(base_path, "dataset", "train-val-test", f"val{DATASET_SIZE_VAL}")
    # IMAGE_SIZE = 256
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_data', dest='source_data', action='store',
                        help='Path to the folder containing subfolders backgrounds/foregrounds')
    parser.add_argument('--dataset_size', dest='dataset_size', action='store',
                        help='Size of COCO synthetic dataset')
    parser.add_argument('--image_size', dest='image_size', action='store',
                        help='Size of the synthetic images as a square')
    parser.add_argument('--split_ratio_train', dest='split_ratio_train', action='store',
                        help='Split size for Training data, remaining is used for validation')

    arguments = parser.parse_args()


    ### CONVERT TO INPUTS FOR WORKFLOW
    dataset_size = int(arguments.dataset_size)
    image_size = int(arguments.image_size)
    split_size = float(arguments.split_ratio_train)
    DATASET_SIZE_TRAIN = int(dataset_size * split_size)

    val_size = 1-split_size
    if val_size > 0:
        DATASET_SIZE_VAL = int(dataset_size * val_size)
    else:
        DATASET_SIZE_VAL = int(dataset_size * .1)

    coco_data_path = Path(arguments.source_data).joinpath('coco_brute_force')
    coco_data_path.mkdir(parents=True, exist_ok=True)

    train_output = coco_data_path.joinpath('train')
    val_output = coco_data_path.joinpath('val')

    generate(train_output, DATASET_SIZE_TRAIN, arguments.source_data, image_size)
    generate(val_output, DATASET_SIZE_VAL, arguments.source_data, image_size)
