# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset
from typing import List, Optional, Union
from mmengine.fileio import get, get_local_path, list_from_file
import os.path as osp




@DATASETS.register_module()
class ExDarkVocDataset(XMLDataset):
    """Dataset for Exdark."""

    METAINFO = {
        'classes':
        ('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People',
        'Table'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0)]
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}')
            xml_path = osp.join(self.sub_data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_ids
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list