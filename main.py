from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QMessageBox
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QUrl, QPoint
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QDesktopServices
from ui import mainWindow_test
import subprocess
import operator
# global params
from ui import InfoNotifier
from ui import UIParamReflect

import qdarkstyle
import os

class Mythread(QThread):
    # 定义信号
    _signal_progress_info = pyqtSignal()

    _signal_button_ctrl = pyqtSignal()

    def __init__(self):
        super(Mythread, self).__init__()
    def run(self):
        while True:
            # 发出信号
            self._signal_progress_info.emit()
            self._signal_button_ctrl.emit()
            # 让程序休眠
            time.sleep(1.5)



if __name__ == "__main__":
    # Fix for linux
    import multiprocessing
    multiprocessing.set_start_method("spawn")

    from core.leras import nn
    nn.initialize_main_env()

    import os
    import sys
    import time
    import argparse

    from core import pathex
    from core import osex
    from pathlib import Path
    from core.interact import interact as io

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def process_extract(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.main( detector                = arguments.detector,
                        input_path              = Path(arguments.input_dir),
                        output_path             = Path(arguments.output_dir),
                        output_debug            = arguments.output_debug,
                        manual_fix              = arguments.manual_fix,
                        manual_output_debug_fix = arguments.manual_output_debug_fix,
                        manual_window_size      = arguments.manual_window_size,
                        face_type               = arguments.face_type,
                        cpu_only                = arguments.cpu_only,
                        force_gpu_idxs          = [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                      )

    p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    p.add_argument('--detector', dest="detector", choices=['s3fd','manual'], default=None, help="Type of detector.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="Writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="Don't writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'full_face_no_align', 'mark_only'], default='full_face', help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU..")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

    p.set_defaults (func=process_extract)

    def process_dev_extract_vggface2_dataset(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.extract_vggface2_dataset( arguments.input_dir,
                                            device_args={'cpu_only'  : arguments.cpu_only,
                                                        'multi_gpu' : arguments.multi_gpu,
                                                        }
                                            )

    p = subparsers.add_parser( "dev_extract_vggface2_dataset", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU.")
    p.set_defaults (func=process_dev_extract_vggface2_dataset)

    def process_dev_extract_umd_csv(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.extract_umd_csv( arguments.input_csv_file,
                                  device_args={'cpu_only'  : arguments.cpu_only,
                                               'multi_gpu' : arguments.multi_gpu,
                                              }
                                )

    p = subparsers.add_parser( "dev_extract_umd_csv", help="")
    p.add_argument('--input-csv-file', required=True, action=fixPathAction, dest="input_csv_file", help="input_csv_file")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU.")
    p.set_defaults (func=process_dev_extract_umd_csv)


    def process_dev_apply_celebamaskhq(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.apply_celebamaskhq( arguments.input_dir )

    p = subparsers.add_parser( "dev_apply_celebamaskhq", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_dev_apply_celebamaskhq)

    def process_dev_test(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.dev_test( arguments.input_dir )

    p = subparsers.add_parser( "dev_test", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_dev_test)

    def process_sort(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main (input_path=Path(arguments.input_dir), sort_by_method=arguments.sort_by_method)

    p = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', dest="sort_by_method", default=None, choices=("blur", "face-yaw", "face-pitch", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "oneface", "final", "absdiff"), help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    p.set_defaults (func=process_sort)

    def process_util(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Util

        if arguments.convert_png_to_jpg:
            Util.convert_png_to_jpg_folder (input_path=arguments.input_dir)

        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)

        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename (input_path=arguments.input_dir)

        #if arguments.remove_fanseg:
        #    Util.remove_fanseg_folder (input_path=arguments.input_dir)

        if arguments.remove_ie_polys:
            Util.remove_ie_polys_folder (input_path=arguments.input_dir)

        if arguments.save_faceset_metadata:
            Util.save_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.restore_faceset_metadata:
            Util.restore_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.pack_faceset:
            io.log_info ("Performing faceset packing...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.pack( Path(arguments.input_dir) )

        if arguments.unpack_faceset:
            io.log_info ("Performing faceset unpacking...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.unpack( Path(arguments.input_dir) )

    p = subparsers.add_parser( "util", help="Utilities.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--convert-png-to-jpg', action="store_true", dest="convert_png_to_jpg", default=False, help="Convert DeepFaceLAB PNG files to JPEG.")
    p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="Add landmarks debug image for aligned faces.")
    p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False, help="Recover original aligned filename.")
    #p.add_argument('--remove-fanseg', action="store_true", dest="remove_fanseg", default=False, help="Remove fanseg mask from aligned faces.")
    p.add_argument('--remove-ie-polys', action="store_true", dest="remove_ie_polys", default=False, help="Remove ie_polys from aligned faces.")
    p.add_argument('--save-faceset-metadata', action="store_true", dest="save_faceset_metadata", default=False, help="Save faceset metadata to file.")
    p.add_argument('--restore-faceset-metadata', action="store_true", dest="restore_faceset_metadata", default=False, help="Restore faceset metadata to file. Image filenames must be the same as used with save.")
    p.add_argument('--pack-faceset', action="store_true", dest="pack_faceset", default=False, help="")
    p.add_argument('--unpack-faceset', action="store_true", dest="unpack_faceset", default=False, help="")

    p.set_defaults (func=process_util)

    def process_train(arguments):
        osex.set_process_lowest_prio()


        kwargs = {'model_class_name'         : arguments.model_name,
                  'saved_models_path'        : Path(arguments.model_dir),
                  'training_data_src_path'   : Path(arguments.training_data_src_dir),
                  'training_data_dst_path'   : Path(arguments.training_data_dst_dir),
                  'pretraining_data_path'    : Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
                  'pretrained_model_path'    : Path(arguments.pretrained_model_dir) if arguments.pretrained_model_dir is not None else None,
                  'no_preview'               : arguments.no_preview,
                  'force_model_name'         : arguments.force_model_name,
                  'force_gpu_idxs'           : arguments.force_gpu_idxs,
                  'cpu_only'                 : arguments.cpu_only,
                  'execute_programs'         : [ [int(x[0]), x[1] ] for x in arguments.execute_program ],
                  'debug'                    : arguments.debug,
                  }
        from mainscripts import Trainer
        Trainer.main(**kwargs)

    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of extracted SRC faceset.")
    p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of extracted DST faceset.")
    p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="Optional dir of extracted faceset that will be used in pretraining mode.")
    p.add_argument('--pretrained-model-dir', action=fixPathAction, dest="pretrained_model_dir", default=None, help="Optional dir of pretrain model files. (Currently only for Quick96).")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
    p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="Disable preview window.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+')
    p.set_defaults (func=process_train)

    def process_merge(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Merger
        Merger.main ( model_class_name       = arguments.model_name,
                      saved_models_path      = Path(arguments.model_dir),
                      training_data_src_path = Path(arguments.training_data_src_dir) if arguments.training_data_src_dir is not None else None,
                      force_model_name       = arguments.force_model_name,
                      input_path             = Path(arguments.input_dir),
                      output_path            = Path(arguments.output_dir),
                      output_mask_path       = Path(arguments.output_mask_dir),
                      aligned_path           = Path(arguments.aligned_dir) if arguments.aligned_dir is not None else None,
                      force_gpu_idxs         = arguments.force_gpu_idxs,
                      cpu_only               = arguments.cpu_only)

    p = subparsers.add_parser( "merge", help="Merger")
    p.add_argument('--training-data-src-dir', action=fixPathAction, dest="training_data_src_dir", default=None, help="(optional, may be required by some models) Dir of extracted SRC faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the merged files will be stored.")
    p.add_argument('--output-mask-dir', required=True, action=fixPathAction, dest="output_mask_dir", help="Output mask directory. This is where the mask files will be stored.")
    p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", default=None, help="Aligned directory. This is where the extracted of dst faces stored.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Merge on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.set_defaults(func=process_merge)

    videoed_parser = subparsers.add_parser( "videoed", help="Video processing.").add_subparsers()

    def process_videoed_extract_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.extract_video (arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="Extract images from video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted images will be stored.")
    p.add_argument('--output-ext', dest="output_ext", default=None, help="Image format (extension) of output files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. 0 - full fps.")
    p.set_defaults(func=process_videoed_extract_video)

    def process_videoed_cut_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.cut_video (arguments.input_file,
                           arguments.from_time,
                           arguments.to_time,
                           arguments.audio_track_id,
                           arguments.bitrate)
    p = videoed_parser.add_parser( "cut-video", help="Cut video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--from-time', dest="from_time", default=None, help="From time, for example 00:00:00.000")
    p.add_argument('--to-time', dest="to_time", default=None, help="To time, for example 00:00:00.000")
    p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None, help="Specify audio track id.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.set_defaults(func=process_videoed_cut_video)

    def process_videoed_denoise_image_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.denoise_image_sequence (arguments.input_dir, arguments.ext, arguments.factor)
    p = videoed_parser.add_parser( "denoise-image-sequence", help="Denoise sequence of images, keeping sharp edges. This allows you to make the final fake more believable, since the neural network is not able to make a detailed skin texture, but it makes the edges quite clear. Therefore, if the whole frame is more `blurred`, then a fake will seem more believable. Especially true for scenes of the film, which are usually very clear.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default=None, help="Image format (extension) of input files.")
    p.add_argument('--factor', type=int, dest="factor", default=None, help="Denoise factor (1-20).")
    p.set_defaults(func=process_videoed_denoise_image_sequence)

    def process_videoed_video_from_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.video_from_sequence (input_dir      = arguments.input_dir,
                                     output_file    = arguments.output_file,
                                     reference_file = arguments.reference_file,
                                     ext      = arguments.ext,
                                     fps      = arguments.fps,
                                     bitrate  = arguments.bitrate,
                                     include_audio = arguments.include_audio,
                                     lossless = arguments.lossless)

    p = videoed_parser.add_parser( "video-from-sequence", help="Make video from image sequence.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--reference-file', action=fixPathAction, dest="reference_file", help="Reference file used to determine proper FPS and transfer audio from it. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default='png', help="Image format (extension) of input files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="FPS of output file. Overwritten by reference-file.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.add_argument('--include-audio', action="store_true", dest="include_audio", default=False, help="Include audio from reference file.")
    p.add_argument('--lossless', action="store_true", dest="lossless", default=False, help="PNG codec.")

    p.set_defaults(func=process_videoed_video_from_sequence)

    def process_labelingtool_edit_mask(arguments):
        from mainscripts import MaskEditorTool
        MaskEditorTool.mask_editor_main (arguments.input_dir, arguments.confirmed_dir, arguments.skipped_dir, no_default_mask=arguments.no_default_mask)

    labeling_parser = subparsers.add_parser( "labelingtool", help="Labeling tool.").add_subparsers()
    p = labeling_parser.add_parser ( "edit_mask", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--confirmed-dir', required=True, action=fixPathAction, dest="confirmed_dir", help="This is where the labeled faces will be stored.")
    p.add_argument('--skipped-dir', required=True, action=fixPathAction, dest="skipped_dir", help="This is where the labeled faces will be stored.")
    p.add_argument('--no-default-mask', action="store_true", dest="no_default_mask", default=False, help="Don't use default mask.")

    p.set_defaults(func=process_labelingtool_edit_mask)

    facesettool_parser = subparsers.add_parser( "facesettool", help="Faceset tools.").add_subparsers()

    def process_faceset_enhancer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetEnhancer
        FacesetEnhancer.process_folder ( Path(arguments.input_dir),
                                         cpu_only=arguments.cpu_only,
                                         force_gpu_idxs=arguments.force_gpu_idxs
                                       )

    p = facesettool_parser.add_parser ("enhance", help="Enhance details in DFL faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Process on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

    p.set_defaults(func=process_faceset_enhancer)

    """
    def process_relight_faceset(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetRelighter
        FacesetRelighter.relight (arguments.input_dir, arguments.lighten, arguments.random_one)

    def process_delete_relighted(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetRelighter
        FacesetRelighter.delete_relighted (arguments.input_dir)

    p = facesettool_parser.add_parser ("relight", help="Synthesize new faces from existing ones by relighting them. With the relighted faces neural network will better reproduce face shadows.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--lighten', action="store_true", dest="lighten", default=None, help="Lighten the faces.")
    p.add_argument('--random-one', action="store_true", dest="random_one", default=None, help="Relight the faces only with one random direction, otherwise relight with all directions.")
    p.set_defaults(func=process_relight_faceset)

    p = facesettool_parser.add_parser ("delete_relighted", help="Delete relighted faces.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.set_defaults(func=process_delete_relighted)
    """

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)


    class My_train_thread(QThread):
        # 定义信号,定义参数为str类型

        def __init__(self):
            super(My_train_thread, self).__init__()

        def run(self):
            self.StartTrain()

        def set_dir(self, dst_align_path = "", src_align_dir = "", models_path = "", pretraining_data_path = "", pretrain_model_path = "", model_save_path = ""):
            self.train_dst_aligned_path = dst_align_path
            self.train_src_aligned_path = src_align_dir
            self.models_path = models_path
            self.pretrain_data_path = pretraining_data_path
            self.pretrain_model_path = pretrain_model_path
            self.model_save_path = model_save_path

        def StartTrain(self):
            osex.set_process_lowest_prio()


            modelName = pathex.get_all_dir_names_startswith(self.models_path, 'Model_')
            if UIParamReflect.UIParam2Config.bUseSAEHD:
                model_class_name = "SAEHD"
                pretrained_model_path = None
            else:
                model_class_name = "Quick96"
                pretrained_model_path = Path(self.pretrain_model_path)

            # gather all model dat files
            model_class_names = ["SAEHD","Quick96"]

            saved_models_path = self.model_save_path #"../../workspace/model"
            saved_models_names_saehd = []
            saved_models_names_quick96 = []
            for filepath in pathex.get_file_paths(saved_models_path):
                filepath_name = filepath.name
                model_name_tmp = ""
                if filepath_name.endswith('SAEHD_data.dat'):
                    model_name_tmp = filepath_name[0:len(filepath_name)-len('_SAEHD_data.dat')]
                    saved_models_names_saehd += [model_name_tmp]
                    InfoNotifier.InfoNotifier.g_progress_info.append(f"高配算法可选模型 {model_name_tmp}")
                if filepath_name.endswith('Quick96_data.dat'):
                    model_name_tmp = filepath_name[0:len(filepath_name) - len('_Quick96_data.dat')]
                    saved_models_names_quick96 += [model_name_tmp]
                    InfoNotifier.InfoNotifier.g_progress_info.append(f"低配算法可选模型 {model_name_tmp}")

            # 处理上一次模型训练但是没有模型的情况
            if UIParamReflect.UIParam2Config.bUseLastModel is True:
                if model_class_name == "SAEHD" and len(saved_models_names_saehd) == 0:
                    InfoNotifier.InfoNotifier.g_progress_info.append("'继续上一次的训练配置'使用高配算法，但没有可用模型! 请使用'新建一个训练模型'创建新模型")
                    UIParamReflect.GlobalConfig.b_training_call_in_progress = False
                    return
                if model_class_name == "Quick96" and len(saved_models_names_quick96) == 0:
                    InfoNotifier.InfoNotifier.g_progress_info.append("'继续上一次的训练配置'使用低配算法，但没有可用模型! 请使用'新建一个训练模型'创建新模型")
                    UIParamReflect.GlobalConfig.b_training_call_in_progress = False
                    return

            # 处理旧模型找到或找不到模型的逻辑
            find_model = False
            if UIParamReflect.UIParam2Config.bUseOldModel is True:
                if model_class_name == "SAEHD":
                    for exist_model in saved_models_names_saehd:
                        if UIParamReflect.UIParam2Config.modelname == exist_model:
                            find_model = True
                            break
                if model_class_name == "Quick96":
                    for exist_model in saved_models_names_quick96:
                        if UIParamReflect.UIParam2Config.modelname == exist_model:
                            find_model = True
                            break

                if find_model is True:
                    InfoNotifier.InfoNotifier.g_progress_info.append(f"找到模型 :{UIParamReflect.UIParam2Config.modelname}, 将继续训练")
                else:
                    InfoNotifier.InfoNotifier.g_progress_info.append(f"输入的模型: {UIParamReflect.UIParam2Config.modelname} 不存在, 请重新输入正确的模型名称")
                    if model_class_name == "SAEHD" and len(saved_models_names_saehd) != 0:
                        InfoNotifier.InfoNotifier.g_progress_info.append("可选旧高配模型为如下：")
                        for exist_model in saved_models_names_saehd:
                            InfoNotifier.InfoNotifier.g_progress_info.append(f"高配算法模型 :{exist_model}")
                    if model_class_name == "Quick96" and len(saved_models_names_quick96) != 0:
                        InfoNotifier.InfoNotifier.g_progress_info.append("可选旧低配模型为如下：")
                        for exist_model in saved_models_names_quick96:
                            InfoNotifier.InfoNotifier.g_progress_info.append(f"低配算法模型 :{exist_model}")
                    UIParamReflect.GlobalConfig.b_training_call_in_progress = False
                    return


            kwargs = {'model_class_name': model_class_name,
                      'saved_models_path': Path(self.model_save_path),
                      'training_data_src_path': Path(self.train_src_aligned_path),
                      'training_data_dst_path': Path(self.train_dst_aligned_path),
                      'pretraining_data_path': Path(self.pretrain_data_path),
                      'pretrained_model_path': pretrained_model_path,
                      'no_preview': False,
                      'force_model_name': None,
                      'force_gpu_idxs': 0,
                      'cpu_only': False,
                      'execute_programs': [],
                      'debug': False,
                      }
            from mainscripts import Trainer
            Trainer.main(**kwargs)

    class My_merge_thread(QThread):
        # 定义信号,定义参数为str类型
        _signal_update_dst_info = pyqtSignal()

        def __init__(self):
            super(My_merge_thread, self).__init__()

        def run(self):
            self.StartMerge()

        def set_dir(self, merge_model_class = "", merge_model_name = "", saved_model_path = "", input_path = "", output_path = "", output_mask_path = "", aligned_path = ""):
            self.merge_model_class = merge_model_class
            self.merge_model_name = merge_model_name
            self.saved_model_path = saved_model_path
            self.input_path = input_path
            self.output_path = output_path
            self.output_mask_path = output_mask_path
            self.aligned_path = aligned_path

        def StartMerge(self):
            osex.set_process_lowest_prio()

            UIParamReflect.UIParam2Config.merge_model_name = self.merge_model_name

            from mainscripts import Merger
            Merger.main(model_class_name=self.merge_model_class,
                        saved_models_path=Path(self.saved_model_path),
                        training_data_src_path=None,
                        force_model_name=None,
                        input_path=Path(self.input_path),
                        output_path=Path(self.output_path),
                        output_mask_path=Path(self.output_mask_path),
                        aligned_path=Path(self.aligned_path),
                        force_gpu_idxs=None,
                        cpu_only=False)

            InfoNotifier.InfoNotifier.g_progress_info.append("-----------------单帧合成图片完毕-----------------")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False
            QApplication.processEvents()

            self._signal_update_dst_info.emit()


    class ApplicationWindow(QMainWindow):

        def __init__(self):
            super(ApplicationWindow, self).__init__()
            self.ui = mainWindow_test.Ui_MainWindow()
            self.ui.setupUi(self)
            self.cwd = os.getcwd()  # 获取当前程序文件位置
            #connect signals
            # thread running
            self.thread = Mythread()
            self.thread._signal_progress_info.connect(self.update_progress_info)
            self.thread._signal_button_ctrl.connect(self.update_button_state)
            self.thread.start()

            self.init_dir()

            UIParamReflect.GlobalConfig.ffmpeg_cmd_path =  self.app_dir + "\\ffmpeg\\ffmpeg"
            UIParamReflect.GlobalConfig.ffprobe_cmd_path = self.app_dir + "\\ffmpeg\\ffprobe"
            # UIParamReflect.GlobalConfig.ffmpeg_cmd_path.replace('/', '\\')
            # UIParamReflect.GlobalConfig.ffprobe_cmd_path.replace('/', '\\')

        def init_dir(self):
            # app dir
            self.app_dir = os.path.abspath(os.path.dirname(__file__))
            # workspace
            self.workspace_dir = self.result_dir = os.path.abspath(os.path.dirname(__file__))+"\\workspace"
            # dst dir
            self.dst_dir = self.workspace_dir + "\\data_dst"
            self.dst_align_dir = self.dst_dir + "\\aligned"
            self.dst_merge_dir = self.dst_dir + "\\merged"
            self.dst_merge_mask_dir = self.dst_dir + "\\merged_mask"
            # src dir
            if self.ui.rb_j3_m_adult.isChecked():       # 成男模型
                self.src_dir = self.workspace_dir + "\\data_src\\m_adult"
            elif self.ui.rb_j3_fm_adult.isChecked():                # 成女模型
                self.src_dir = self.workspace_dir + "\\data_src\\fm_adult"
            elif self.ui.rb_j3_fm.isChecked():          # 萝莉模型
                self.src_dir = self.workspace_dir + "\\data_src\\fm"
            elif self.ui.rb_j3_m.isChecked():           # 正太模型
                self.src_dir = self.workspace_dir + "\\data_src\\m"
            self.src_align_dir = self.src_dir+"\\aligned"

            self.pretrained_data_path = self.app_dir + "\\pretrain_CelebA"
            self.pretrained_model_path= self.app_dir + "\\pretrain_Quick96"

            self.models_path = self.app_dir + "\\models"
            self.model_save_path = self.workspace_dir + "\\model"

            self.update_ui_dst_info()
            self.update_ui_src_info()

        def validate(self, type = ""): # type "train" "mergePic" "mergeVideo"
            dst_align_file_count = 0
            src_align_file_count = 0
            dst_merge_file_count = 0

            for filepath in pathex.get_file_paths(self.dst_align_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png') or filepath_name.endswith('.jpg'):
                    dst_align_file_count +=1

            for filepath in pathex.get_file_paths(self.src_align_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png') or filepath_name.endswith('.jpg'):
                    src_align_file_count +=1

            for filepath in pathex.get_file_paths(self.dst_merge_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png') or filepath_name.endswith('.jpg'):
                    dst_merge_file_count +=1

            if type is "train":
                if dst_align_file_count is 0 and src_align_file_count is 0:
                    reply = QMessageBox.information(self,   "操作提示", "不存在可训练的数据，请检查data_dst/align和data_src/align目录",QMessageBox.Ok)
                    return False

            if type is "mergePic":
                 if self.ui.le_model_path.text() is "":
                    reply = QMessageBox.information(self, "操作提示", "无法合成，请先选择换脸所需要的模型!",QMessageBox.Ok)
                    return False

            if type is "mergeVideo":
                if dst_merge_file_count is 0:
                    reply = QMessageBox.information(self, "操作提示", "无法合成视频，请先逐帧生成图片!",QMessageBox.Ok)
                    return False

            return True

        def update_ui_dst_info(self):
            self.ui.le_dst_merge_dir.setText(self.dst_merge_dir)
            self.ui.le_result_dir.setText(self.result_dir)

            # 处理目标视频是否有提取关键帧数据
            dst_file_png_count = 0
            dst_file_jpg_count = 0
            for filepath in pathex.get_file_paths(self.dst_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png'):
                    dst_file_png_count +=1
                if filepath_name.endswith('.jpg'):
                    dst_file_jpg_count += 1

            if dst_file_png_count != 0 or dst_file_jpg_count !=0 :
                self.ui.label_dst_suggest.setText(f"data_dst目录存在图片{dst_file_png_count+dst_file_jpg_count}个，可以'开始提取人脸' ")
            else:
                self.ui.label_dst_suggest.setText("data_dst目录不存在数据，请先'开始提取图像'")

            # 处理目标视频帧是否有提取对齐后的人脸
            dst_file_png_count = 0
            dst_file_jpg_count = 0
            for filepath in pathex.get_file_paths(self.dst_align_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png'):
                    dst_file_png_count +=1
                if filepath_name.endswith('.jpg'):
                    dst_file_jpg_count += 1

            if dst_file_png_count != 0 or dst_file_jpg_count !=0 :
                self.ui.label_dst_aligned_suggest.setText(f"data_dst/aligned目录已有图片 {dst_file_png_count+dst_file_jpg_count}个，可以进行训练")
            else:
                self.ui.label_dst_aligned_suggest.setText("data_dst/aligned目录不存在数据，请先'开始提取人脸'")

            # 合并替换图像
            dst_file_png_count = 0
            dst_file_jpg_count = 0
            for filepath in pathex.get_file_paths(self.dst_merge_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png'):
                    dst_file_png_count +=1
                if filepath_name.endswith('.jpg'):
                    dst_file_jpg_count += 1
            self.ui.label_dst_merge_file_count.setText("共有%d个png文件，%d个jpg文件"%(dst_file_png_count,dst_file_jpg_count))
            if dst_file_png_count != 0 or dst_file_jpg_count !=0 :
                self.ui.label_dst_merge_suggest.setText("已存在视频帧的替换结果，可以进行'合成视频' ")
            else:
                self.ui.label_dst_merge_suggest.setText("不存在视频帧替换的数据，请先'逐帧合成'")
            QApplication.processEvents()

        def update_src(self):
            if self.ui.rb_j3_m_adult.isChecked():       # 成男模型
                self.src_dir = self.workspace_dir + "\\data_src\\m_adult"
            elif self.ui.rb_j3_fm_adult.isChecked():                # 成女模型
                self.src_dir = self.workspace_dir + "\\data_src\\fm_adult"
            elif self.ui.rb_j3_fm.isChecked():          # 萝莉模型
                self.src_dir = self.workspace_dir + "\\data_src\\fm"
            elif self.ui.rb_j3_m.isChecked():           # 正太模型
                self.src_dir = self.workspace_dir + "\\data_src\\m"
            self.src_align_dir = self.src_dir+"\\aligned"

            self.update_ui_src_info()

        def update_ui_src_info(self):
            # 处理剑三是否存在原始数据
            self.src_file_png_count = 0
            self.src_file_jpg_count = 0
            for filepath in pathex.get_file_paths(self.src_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png'):
                    self.src_file_png_count +=1
                if filepath_name.endswith('.jpg'):
                    self.src_file_jpg_count += 1
            self.ui.le_src_dir.setText(self.src_dir)
            self.ui.label_src_file_count.setText("共有%d个图像文件"%(self.src_file_png_count+self.src_file_jpg_count))
            if self.src_file_png_count != 0 or self.src_file_jpg_count !=0 :
                self.ui.label_src_suggest.setText("data_src目录存在剑三原始数据，可以 '开始提取人脸' ")
            else:
                self.ui.label_src_suggest.setText("data_src目录不存在剑三原始数据，请检查数据!")

            # 处理剑三是否有已经提取的人脸数据
            self.src_align_file_png_count = 0
            self.src_align_file_jpg_count = 0
            for filepath in pathex.get_file_paths(self.src_align_dir):
                filepath_name = filepath.name
                if filepath_name.endswith('.png'):
                    self.src_align_file_png_count +=1
                if filepath_name.endswith('.jpg'):
                    self.src_align_file_jpg_count += 1
            self.ui.le_src_align_dir.setText(self.src_align_dir)
            self.ui.label_src_align_file_count.setText("共有%d个图像文件"%(self.src_align_file_png_count+self.src_align_file_jpg_count))
            if self.src_align_file_png_count != 0 or self.src_align_file_jpg_count !=0 :
                self.ui.label_src_align_suggest.setText("data_src/aligned目录存在人脸资源，继续提取将会覆盖原始文件")
            else:
                self.ui.label_src_align_suggest.setText("data_src/aligned目录不存在剑三人脸资源，提取先请确认原始资源存在!")

        def update_progress_info(self):
            for info in InfoNotifier.InfoNotifier.g_progress_info:
                self.ui.progress_view.append(info)
            InfoNotifier.InfoNotifier.g_progress_info.clear()

        def update_ui_progress_info(self, info = ""):
            self.ui.progress_view.append(info)

        def update_button_state(self):
            if UIParamReflect.GlobalConfig.b_sync_block_op_in_progress is True: # 全局进行一个同步阻塞操作
                self.ui.pushButton_5.setEnabled(False)
                self.ui.pb_open_data_dst_dir.setEnabled(False)
                self.ui.pushButton_3.setEnabled(False)
                self.ui.pushButton_4.setEnabled(False)
                self.ui.pushButton.setEnabled(False)
                self.ui.pb_save.setEnabled(False)
                self.ui.pb_end.setEnabled(False)
                self.ui.pushButton_6.setEnabled(False)
                self.ui.pushButton_7.setEnabled(False)
                self.ui.pushButton_2.setEnabled(False)
            else:
                self.ui.pushButton_5.setEnabled(True)
                self.ui.pb_open_data_dst_dir.setEnabled(True)
                self.ui.pushButton_3.setEnabled(True)
                self.ui.pushButton_4.setEnabled(True)
                self.ui.pushButton.setEnabled(True)
                self.ui.pb_save.setEnabled(True)
                self.ui.pb_end.setEnabled(True)
                self.ui.pushButton_6.setEnabled(True)
                self.ui.pushButton_7.setEnabled(True)
                self.ui.pushButton_2.setEnabled(True)

                if UIParamReflect.GlobalConfig.b_training_call_in_progress is True:
                    self.ui.pushButton.setEnabled(False)
                    self.ui.pb_save.setEnabled(False)
                    self.ui.pb_end.setEnabled(False)
                else:
                    self.ui.pushButton.setEnabled(True)
                    self.ui.pb_save.setEnabled(True)
                    self.ui.pb_end.setEnabled(True)

                    if UIParamReflect.UIParam2Config.train_state == 0:
                        self.ui.pushButton.setEnabled(True)
                        self.ui.pb_save.setEnabled(False)
                        self.ui.pb_end.setEnabled(False)
                    elif UIParamReflect.UIParam2Config.train_state == 1:
                        self.ui.pushButton.setEnabled(False)
                        self.ui.pb_save.setEnabled(True)
                        self.ui.pb_end.setEnabled(True)

        def StartTrain(self):
            if self.validate(type = "train") is False:
                return

            self.update_ui_progress_info("\n-----------------开始训练-----------------")
            QApplication.processEvents()
            self.update_src()

            self.trainThread = My_train_thread()
            self.trainThread.set_dir(self.dst_align_dir, self.src_align_dir, self.models_path, self.pretrained_data_path
                                     , self.pretrained_model_path, self.model_save_path)
            self.trainThread.start()

            UIParamReflect.GlobalConfig.b_training_call_in_progress = True

        def modelSelect(self):
            if self.ui.rb_use_SAE.isChecked():
                if self.ui.rb_last_model_sae.isChecked():
                    UIParamReflect.UIParam2Config.bUseLastModel = True
                else:
                    UIParamReflect.UIParam2Config.bUseLastModel = False

            if self.ui.rb_use_Quick96.isChecked():
                if self.ui.rb_last_model_quick96.isChecked():
                    UIParamReflect.UIParam2Config.bUseLastModel = True
                else:
                    UIParamReflect.UIParam2Config.bUseLastModel = False

            # use new model
            if self.ui.rb_use_SAE.isChecked():
                if self.ui.rb_new_model_sae.isChecked():
                    UIParamReflect.UIParam2Config.bUseNewModel = True
                    self.ui.le_new_model_name_sae.setEnabled(True)
                    self.ui.rb_device_CPU_sae.setEnabled(True)
                    self.ui.rb_device_GPU_sae.setEnabled(True)
                else:
                    UIParamReflect.UIParam2Config.bUseNewModel = False
                    self.ui.le_new_model_name_sae.setEnabled(False)
                    self.ui.rb_device_CPU_sae.setEnabled(False)
                    self.ui.rb_device_GPU_sae.setEnabled(False)

            if self.ui.rb_use_Quick96.isChecked():
                if self.ui.rb_new_model_quick96.isChecked():
                    UIParamReflect.UIParam2Config.bUseNewModel = True
                    self.ui.le_new_model_name_quick96.setEnabled(True)
                    self.ui.rb_device_CPU_quick96.setEnabled(True)
                    self.ui.rb_device_GPU_quick96.setEnabled(True)
                else:
                    UIParamReflect.UIParam2Config.bUseNewModel = False
                    self.ui.le_new_model_name_quick96.setEnabled(False)
                    self.ui.rb_device_CPU_quick96.setEnabled(False)
                    self.ui.rb_device_GPU_quick96.setEnabled(False)

            # old model
            if self.ui.rb_use_SAE.isChecked():
                if self.ui.rb_old_model_sae.isChecked():
                    UIParamReflect.UIParam2Config.bUseOldModel = True
                    self.ui.le_old_model_name_sae.setEnabled(True)
                else:
                    UIParamReflect.UIParam2Config.bUseOldModel = False
                    self.ui.le_old_model_name_sae.setEnabled(False)

            if self.ui.rb_use_Quick96.isChecked():
                if self.ui.rb_old_model_quick96.isChecked():
                    UIParamReflect.UIParam2Config.bUseOldModel = True
                    self.ui.le_old_model_name_quick96.setEnabled(True)
                else:
                    UIParamReflect.UIParam2Config.bUseOldModel = False
                    self.ui.le_old_model_name_quick96.setEnabled(False)

        def modelNameEdit(self):
            if self.ui.rb_use_SAE.isChecked():
                if self.ui.rb_new_model_sae.isChecked():
                    UIParamReflect.UIParam2Config.modelname = self.ui.le_new_model_name_sae.text()
                if self.ui.rb_old_model_sae.isChecked():
                    UIParamReflect.UIParam2Config.modelname = self.ui.le_old_model_name_sae.text()
            if self.ui.rb_use_Quick96.isChecked():
                if self.ui.rb_new_model_quick96.isChecked():
                    UIParamReflect.UIParam2Config.modelname = self.ui.le_new_model_name_quick96.text()
                if self.ui.rb_old_model_quick96.isChecked():
                    UIParamReflect.UIParam2Config.modelname = self.ui.le_old_model_name_quick96.text()

        def AlgoSelect(self):
            if self.ui.rb_use_SAE.isChecked():
                UIParamReflect.UIParam2Config.bUseSAEHD = True
                self.ui.rb_last_model_sae.setEnabled(True)
                self.ui.rb_new_model_sae.setEnabled(True)
                self.ui.rb_old_model_sae.setEnabled(True)

                self.ui.rb_last_model_quick96.setEnabled(False)
                self.ui.rb_new_model_quick96.setEnabled(False)
                self.ui.rb_old_model_quick96.setEnabled(False)
                self.ui.le_old_model_name_quick96.setEnabled(False)
                self.ui.le_new_model_name_quick96.setEnabled(False)
                self.ui.rb_device_CPU_quick96.setEnabled(False)
                self.ui.rb_device_GPU_quick96.setEnabled(False)
                self.modelSelect()
            else:
                UIParamReflect.UIParam2Config.bUseSAEHD = False
                self.ui.rb_last_model_quick96.setEnabled(True)
                self.ui.rb_new_model_quick96.setEnabled(True)
                self.ui.rb_old_model_quick96.setEnabled(True)

                self.ui.rb_last_model_sae.setEnabled(False)
                self.ui.rb_new_model_sae.setEnabled(False)
                self.ui.rb_old_model_sae.setEnabled(False)
                self.ui.le_old_model_name_sae.setEnabled(False)
                self.ui.le_new_model_name_sae.setEnabled(False)
                self.ui.rb_device_CPU_sae.setEnabled(False)
                self.ui.rb_device_GPU_sae.setEnabled(False)
                self.modelSelect()

        def DeviceSelect(self):
            if self.ui.rb_device_CPU_sae.isEnabled() and self.ui.rb_device_GPU_sae.isEnabled():
                if self.ui.rb_device_CPU_sae.isChecked():
                    UIParamReflect.UIParam2Config.bUseGPU = False
                else:
                    UIParamReflect.UIParam2Config.bUseGPU = True

            if self.ui.rb_device_CPU_quick96.isEnabled() and self.ui.rb_device_GPU_quick96.isEnabled():
                if self.ui.rb_device_CPU_quick96.isChecked():
                    UIParamReflect.UIParam2Config.bUseGPU = False
                else:
                    UIParamReflect.UIParam2Config.bUseGPU = True

        def SaveOnce(self):
            self.update_ui_progress_info("\n-----------------开始保存训练结果-----------------")
            QApplication.processEvents()
            UIParamReflect.UIParam2Config.bSaveOnce = True
            UIParamReflect.GlobalConfig.b_training_call_in_progress = True

        def StopTrain(self):
            self.update_ui_progress_info("\n-----------------结束训练-----------------")
            QApplication.processEvents()
            UIParamReflect.UIParam2Config.bStopTrain = True
            UIParamReflect.GlobalConfig.b_training_call_in_progress = True

        # step 1 extract video data_dst
        def extract_dst_video_real(self, input_path = "", output_path = "", output_ext = "", fps = 0):
            osex.set_process_lowest_prio()
            from mainscripts import VideoEd
            VideoEd.extract_video(input_path, output_path, output_ext, fps)

        ### 从目标视频中提取图像 ###
        def extract_dst_video(self):
            input_file = self.ui.le_data_dst_path.text()

            if input_file is "":
                InfoNotifier.InfoNotifier.g_progress_info.append("没有设置目标视频路径，请先选择目标视频!")
                return

            output_dir = self.dst_dir
            self.ori_dst_video_path = input_file
            self.update_ui_progress_info("\n图像提取路径:" + output_dir)

            self.update_ui_progress_info("\n图像输出路径:" + self.dst_dir)

            # if self.ui.rb_export_png.isChecked():
            #  output_ext = "png"
            # if self.ui.rb_export_jpg.isChecked():
            #  output_ext = "jpg"
            output_ext = "jpg"
            self.update_ui_progress_info("\n图像格式:" + output_ext)

            fps = 0
            self.update_ui_progress_info("\n-----------------开始提取视频帧-----------------")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
            QApplication.processEvents()
            self.extract_dst_video_real(input_file, output_dir, output_ext, fps)
            self.update_ui_progress_info("\n-----------------提取视频帧完毕-----------------")

            #更新一下信息
            self.update_ui_dst_info()
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False

        ### 从文件系统中选择一个视频文件 ###
        def ChooseDstDir(self):

            fileName_choose, filetype = QFileDialog.getOpenFileName(self, "选取文件",
                                                                    self.cwd,  # 起始路径
                                                                    "MP4 Files (*.mp4);;AVI Files (*.avi);;FLV Files (*.flv)")  # 设置文件扩展名过滤,用双分号间隔
            if fileName_choose == "":
                print("\n取消选择")
                return
            self.ui.le_data_dst_path.setText(fileName_choose)
            InfoNotifier.InfoNotifier.g_progress_info.append("选择视频路径:" + fileName_choose)


        ### 从文件系统中选择一个模型文件 ###
        def ChooseModelDir(self):

            model_path_choose, filetype = QFileDialog.getOpenFileName(self, "选取文件",
                                                                    self.cwd + "\\workspace\\model",  # 起始路径
                                                                    "model Files (*_data.dat)")  # 设置文件扩展名过滤,用双分号间隔
            if model_path_choose == "":
                print("\n取消选择")
                return
            self.ui.le_model_path.setText(model_path_choose)
            InfoNotifier.InfoNotifier.g_progress_info.append("选择模型路径:" + model_path_choose)

            model_file_name = os.path.basename(model_path_choose)


            if model_file_name.endswith('Quick96_data.dat'):
                self.merge_model_class = "Quick96"
                self.merge_model_name = model_file_name[0:len(model_file_name) - len('_Quick96_data.dat')]
            if model_file_name.endswith('SAEHD_data.dat'):
                self.merge_model_class = "SAEHD"
                self.merge_model_name = model_file_name[0:len(model_file_name) - len('_SAEHD_data.dat')]


        def EditDstDir(self):
            dir_path = Path(self.ui.le_data_dst_path.text())
            if dir_path.exists() == False:
                print("\n文件不存在")
                return

        def extract_full_face(self):
            self.update_ui_progress_info("-----------------开始提取人脸图像------------------")
            self.update_ui_progress_info(f"人脸图像提取路径: {self.dst_dir}")
            self.update_ui_progress_info(f"人脸图像输出路径: {self.dst_align_dir}")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
            QApplication.processEvents()
            osex.set_process_lowest_prio()
            from mainscripts import Extractor
            Extractor.main(detector='s3fd',
                           input_path=Path(self.dst_dir),
                           output_path=Path(self.dst_align_dir),
                           output_debug=False,
                           manual_fix=False,
                           manual_output_debug_fix=False,
                           manual_window_size=1368,
                           face_type='full_face',
                           cpu_only=False,
                           force_gpu_idxs= None,
                           )
            self.update_ui_progress_info("-----------------人脸图像提取完毕-----------------")
            #更新一下信息
            self.update_ui_dst_info()
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False

        def extract_j3_face(self):
            self.update_ui_progress_info("-----------------开始提取剑三人脸图像-----------------")
            self.update_src()
            self.update_ui_progress_info(f"人脸图像提取路径: {self.src_dir}")
            if self.src_file_png_count == 0 and self.src_file_jpg_count == 0 :
                self.update_ui_progress_info("人脸图像提取路径下不存在剑三原始数据，请先确认数据存在，再继续操作！")
                return

            self.update_ui_progress_info(f"人脸图像输出路径: {self.src_align_dir}")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
            QApplication.processEvents()


            from mainscripts import Extractor
            Extractor.main(detector='s3fd',
                           input_path=Path(self.src_dir),
                           output_path=Path(self.src_align_dir),
                           output_debug=False,
                           manual_fix=False,
                           manual_output_debug_fix=False,
                           manual_window_size=1368,
                           face_type='full_face',
                           cpu_only=False,
                           force_gpu_idxs= None,
                           )
            self.update_ui_progress_info("-----------------人脸剑三图像提取完毕-----------------")
            #更新一下信息
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False
            self.update_ui_src_info()


        def merge_src_to_dst(self):
            if self.validate(type = "mergePic") is False:
                return

            self.update_ui_progress_info("\n-----------------开始逐帧替换人脸-----------------")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
            QApplication.processEvents()
            self.mergeThread = My_merge_thread()
            self.mergeThread._signal_update_dst_info.connect(self.update_ui_dst_info)

            self.mergeThread.set_dir(self.merge_model_class, self.merge_model_name, self.model_save_path, self.dst_dir, self.dst_merge_dir, self.dst_merge_mask_dir,
                                     self.dst_align_dir)

            self.mergeThread.start()

        def merge_to_mp4(self):
            if self.validate(type = "mergeVideo") is False:
                return
            
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = True
            self.update_ui_progress_info("-----------------开始合成最终视频-----------------")
            QApplication.processEvents()
            osex.set_process_lowest_prio()
            from mainscripts import VideoEd
            VideoEd.video_from_sequence (input_dir      = Path(self.dst_merge_dir),
                                         output_file    = Path(self.workspace_dir + "/result.mp4"),
                                         reference_file = None, # Path(self.workspace_dir + "/data_dst.mp4"),  #need ori video path
                                         ext      = "jpg",  #need ori image format
                                         fps      = None,
                                         bitrate  = None,
                                         include_audio = False,
                                         lossless = False)
            self.update_ui_progress_info("-----------------视频合成完毕-----------------")
            output_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "\\workspace\\result.mp4"
            self.update_ui_progress_info(f"视频输出路径为:{output_path} ")
            UIParamReflect.GlobalConfig.b_sync_block_op_in_progress = False
            QApplication.processEvents()

        def merge_result(self):
            self.merge_src_to_dst()

        def merge_video(self):
            self.merge_to_mp4()

        def go_to_src(self, pos):
            menu = QMenu()
            opt = menu.addAction("打开所在文件夹")
            action = menu.exec_(self.ui.le_src_dir.mapToGlobal(pos))
            if action == opt:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.src_dir))

        def go_to_src_align(self, pos):
            menu = QMenu()
            opt = menu.addAction("打开所在文件夹")
            action = menu.exec_(self.ui.le_src_align_dir.mapToGlobal(pos))
            if action == opt:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.src_align_dir))

        def go_to_dst_merge(self, pos):
            menu = QMenu()
            opt = menu.addAction("打开所在文件夹")
            action = menu.exec_(self.ui.le_dst_merge_dir.mapToGlobal(pos))
            if action == opt:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.dst_merge_dir))

        def go_to_result(self, pos):
            menu = QMenu()
            opt = menu.addAction("打开所在文件夹")
            action = menu.exec_(self.ui.le_result_dir.mapToGlobal(pos))
            if action == opt:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self.result_dir))


    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    icon = QIcon()
    icon.addPixmap(QtGui.QPixmap("ui\\icons.tga"),QtGui.QIcon.Normal, QtGui.QIcon.Off)

    window = ApplicationWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.setWindowIcon(icon)
    window.setFixedSize(1060,543)
    window.show()

    sys.exit(app.exec_())

    #arguments = parser.parse_args()
    #arguments.func(arguments)

    print ("Done.")

'''
import code
code.interact(local=dict(globals(), **locals()))
'''
