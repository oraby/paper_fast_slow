from ..common.plottracesprocessor import PlotTraces
from ..common.clr import colorMapParula, colorMapFireFiji
from ..pipeline.pipeline import (DFProcessor, getRowTracesSets,
                                 createRowTracesSet)
try:
    from wfield.utils import reconstruct
except ModuleNotFoundError:
    import sys
    print("wfield package (https://github.com/jcouto/wfield) not found",
          file=sys.stderr)
import cv2
import tifffile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from math import gcd
import io
from pathlib import Path
from typing import List

Cols = List[DFProcessor]

class UnifyBilateralRegionsTraces(DFProcessor):
    def __init__(self, info_df_dict):
        '''Combine the left and right hemispheres brain-regions into a single
        brain-region.
        When computing the unified value, the weight of each brain-region within
        a hemisphere is a a function of its number of valid pixels,
        i.e the part of brain-region within our FOV.
        '''
        self._info_df_dict = info_df_dict

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_traces_dict = getRowTracesSets(row)
            traces_dict = set_name_to_traces_dict["neuronal"]
            traces_keys = set([key.rsplit("_left", 1)[0].rsplit("_right", 1)[0]
                               for key in traces_dict.keys()])
            new_traces_dict = {}
            info_df = self._info_df_dict[row.info_df_id]
            for key_stripped in traces_keys:
                left_name = f"{key_stripped}_left"
                right_name = f"{key_stripped}_right"
                left_trace = traces_dict.get(left_name)
                right_trace = traces_dict.get(right_name)
                if left_trace is None:
                    assert right_trace is not None
                    new_traces_dict[key_stripped] = right_trace.copy()
                elif right_trace is None:
                    new_traces_dict[key_stripped] = left_trace.copy()
                else:
                    left_row = info_df[info_df.area_name == left_name]
                    right_row = info_df[info_df.area_name == right_name]
                    assert len(left_row) == 1, (
                        "Expected one left row for area "
                        f"{left_name} but found: {len(left_row)} - "
                        "are you passing updated info dict?")
                    assert len(right_row) == 1, (
                        "Expected one right row for area "
                        f"{right_name} but found: {len(right_row)} - "
                        "are you passing updated info dict?")
                    left_weight = len(left_row.iloc[0].area_valid_pix)
                    right_weight = len(right_row.iloc[0].area_valid_pix)
                    areas_avg = (
                           (left_trace*left_weight + right_trace*right_weight) /
                           (left_weight+right_weight))
                    new_traces_dict[key_stripped] = areas_avg
            # Create a new dict, I"m not sure whether this step is necessary
            set_name_to_traces_dict = set_name_to_traces_dict.copy()
            set_name_to_traces_dict["neuronal"] = new_traces_dict
            row = createRowTracesSet(row, set_name_to_traces_dict)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self) -> str:
        return (f"Unify left and right brain areas traces into one area")

class WholeBrainActivity(DFProcessor):
    def __init__(self, info_df_dict, relative_to_avg=False):
        self._info_df_dict = info_df_dict
        if relative_to_avg:
            raise NotImplementedError()

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_traces_dict = getRowTracesSets(row)
            traces_dict = set_name_to_traces_dict["neuronal"]

            new_traces_dict = traces_dict.copy()
            for _dir in ["_left", "_right"]:
                weighted_traces = []
                total_weights = 0
                info_df = self._info_df_dict[row.info_df_id]
                total_valid_pix = []
                total_invalid_pix = []
                for trace_id, trace_data in traces_dict.items():
                    area_info = info_df[info_df.area_name == trace_id]
                    assert len(area_info) == 1, (f"Can't find ID {trace_id}, "
                        "make sure you run WholeBrainActivity activity "
                        "processor before UnifyBilateralRegionsTraces")
                    if not trace_id.endswith(_dir):
                        continue
                    valid_pix = area_info.iloc[0].area_valid_pix
                    trace_weight = len(valid_pix)
                    weighted_traces.append(trace_data*trace_weight)
                    total_weights += trace_weight
                    # Now store the pixels so we can draw them later if needed
                    total_valid_pix.append(valid_pix)
                    total_invalid_pix.append(area_info.iloc[0].area_invalid_pix)
                new_traces_dict[f"Crtx{_dir}"] = \
                         np.sum(np.array(weighted_traces), axis=0)/total_weights
                assert new_traces_dict[f"Crtx{_dir}"].shape == trace_data.shape
                # Update the df entry if it exists, otherwise add it
                fields = {"name": "Cortex",
                          "acronym": "Crtx",
                          "area_name": f"Crtx{_dir}",
                          "label": 999,
                          "area_number": -999 if _dir == "_right" else 999,
                          "area_valid_pix": [np.concatenate(total_valid_pix)],
                          "area_invalid_pix": [
                                             np.concatenate(total_invalid_pix)],
                          "allen_rgb": [[0, 0, 0]]
                           # TODO: Fill rest of the fields,
                }
                crtx_row = info_df[info_df.area_name == fields["area_name"]]
                if len(crtx_row):
                    indx = crtx_row.index
                    # fields = {key:[val] for key, val in fields.items()}
                else: # We can use pandas's 'Setting With Enlargement'
                    indx = max(info_df.index) + 1
                for key, val in fields.items():
                    if isinstance(val, list):
                        val = [val] # I don't know why it work but it does
                    info_df.loc[indx, key] = val
            set_name_to_traces_dict = set_name_to_traces_dict.copy()
            set_name_to_traces_dict["neuronal"] = new_traces_dict
            row = createRowTracesSet(row, set_name_to_traces_dict)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self) -> str:
        return "Computing average cortex activity"


class GenVid(DFProcessor):
    def __init__(self, assign_mov : bool, save_prefix=None, save_tiff=True,
                overlay_text=True, save_raw=True):
        assert assign_mov is not None or save_prefix is not None, (
            "Must specify assign_mov or save_prefix parameters")
        self._assign_mov = assign_mov
        self._save_tiff = save_tiff
        if not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._overlay_text = overlay_text
        self._save_raw = save_raw

    def process(self, data):
        rows = data.apply(self._handleRow, axis=1)
        data = pd.DataFrame(rows)
        return data

    def _handleRow(self, row):
        U = np.load(row.U)
        traces_vals = row.traces_sets["neuronal"].values()
        traces_min = min([trace.min() for trace in traces_vals])
        traces_max = max([trace.max() for trace in traces_vals])
        mov, mov_float = self._buildAvgSVTMovie(row, U)
        fps = row.acq_sampling_rate
        if self._overlay_text:
            frame_num_to_epoch_name = {}
            for rng, epoch in zip(row.epochs_ranges, row.epochs_names):

                frame_num_to_epoch_name.update(
                               # idx+1: Frames start from 1
                               {idx+1:epoch for idx in range(rng[0], rng[1]+1)})
            mov = self._overlayEpochs(mov, frame_num_to_epoch_name, fps)
        if self._savePrefix is not None:
            save_fp = self._savePrefix(plot_id=row.TrialNumber,
                                       session_name=row.ShortName,
                                       df=pd.DataFrame([row]))
            if self._save_raw:
                np.save(save_fp.with_suffix('.npy'), mov_float)

            if self._save_tiff:
                if isinstance(save_fp, str):
                    save_fp = Path(save_fp)
                save_fp_tiff = save_fp.with_suffix('.tiff')
                print("save_fp_tiff:", save_fp_tiff)
                _saveTiff(mov_float, # mov,
                          save_fp=save_fp_tiff, traces_min=traces_min,
                          traces_max=traces_max)
            _saveMovie(mov, fps=fps, save_fp=save_fp)
        if self._assign_mov:
            row = row.copy()
            row["mov"] = mov # TODO: Assign as part of the neuronal set
        return row

    def _buildAvgSVTMovie(self, row, U):
        set_name_to_traces_dict = getRowTracesSets(row)
        svt_trace = set_name_to_traces_dict["SVT_MOV"]["movie"]
        print("Reconstructing movie...")
        avg_movie_float = reconstruct(u=U, svt=svt_trace)#[:,:10])
        # avg_movie *= 255/np.amax(avg_movie)
        avg_min, avg_max = avg_movie_float.min(), avg_movie_float.max()
        clip = False
        if row.ShortName == "WF3_M11":
            avg_min, avg_max = -500.1884, 826.2079
            clip = True
        elif row.ShortName == "WF4_M13":
            avg_min, avg_max = -291.03003, 558.05646
            clip = True
        print("mov min max:", row.ShortName, avg_min, avg_max)
        # Sets the range from 0 to 1:
        avg_movie = (avg_movie_float - avg_min) / (avg_max - avg_min)
        if clip:
            avg_movie[avg_movie > 1] = 1
            avg_movie[avg_movie < 0] = 0
        # Sets the range from 0 to 255
        avg_movie = (avg_movie * 255).astype(np.uint8)
        print("Reconstructed movie shape:", avg_movie.shape,
              "- dtype:", avg_movie.dtype)
        # Convert to colored images
        avg_movie_color = [cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2RGB)
                           for gray_frame in avg_movie]
        avg_movie_color = np.array(avg_movie_color)
        # plt.imshow(avg_movie_color[0])
        # plt.show()
        # print("U.shape:", U.shape,
        #       "avg_movie_color.shape:", avg_movie_color.shape,
        #       "avg_movie_float.shape:", avg_movie_float.shape)
        return avg_movie_color, avg_movie_float

    def _overlayEpochs(self, mov, frame_num_to_epoch_name, fps):
        # avg_movie_color
        from PIL import Image, ImageDraw
        res_mov = []
        for i, frame_color in enumerate(tqdm(mov,
                                            desc="Placing plot next to movie")):
            epoch_name = frame_num_to_epoch_name.get(i+1, "*UNKOWN")
            img = Image.fromarray(frame_color)
            img_draw = ImageDraw.Draw(img)
            img_draw.text((5, frame_color.shape[1] - 5),
                          f"{epoch_name} - Sec: {i/fps:.1f}")
            res_mov.append(np.asarray(img))
        res_mov = np.array(res_mov)
        assert mov.dtype == res_mov.dtype, (
            f"{mov.dtype = } != {res_mov.dtyoe = }")
        return res_mov

    def _defaultSavePrefix(self, plot_id, session_name, df=None):
        save_prefix = f"{self._save_prefix}/{session_name}/{plot_id}.avi"
        return save_prefix

    def descr(self):
        return "Generating movie"

class SaveActivityPloltVid(DFProcessor):
    def __init__(self, is_avg_trc, info_df_dict, areas_to_colors, dpi,
                 only_traces_ids=[], save_tiff=True, save_prefix=None,
                 plot_y_lims=None, plot_max_x=None, plot_fig_size=None,
                 sess_avi_min_max={}):
        self._only_traces_ids = only_traces_ids
        self._info_df_dict = info_df_dict
        self._areas_to_colors = areas_to_colors
        self._plotter = PlotTraces(is_avg_trc=is_avg_trc,
                                   areas_to_colors=areas_to_colors,
                                   only_traces_ids=only_traces_ids,
                                   stop_after=1, fig_size=plot_fig_size,
                                   y_lims=plot_y_lims, max_x=plot_max_x)
        self._save_tiff = save_tiff
        if not callable(save_prefix):
            self._save_prefix = save_prefix
            save_prefix = self._defaultSavePrefix
        self._savePrefix = save_prefix
        self._dpi = dpi
        self._gen_vid = GenVid(assign_mov=True, overlay_text=False,
                               save_prefix=save_prefix, save_tiff=False)
        self._sess_avi_min_max = sess_avi_min_max

    def process(self, data):
        assert len(data) == 1, "Expected a single concatenated epochs row"
        ax_plt = self._plotter._plotTraces(data) # Plotter needs a dataframe
        row = data.iloc[0]
        fps = row.acq_sampling_rate
        avg_movie_color = self._gen_vid.process(data).mov.iloc[0]
        avg_movie_w_h = avg_movie_color.shape[2], avg_movie_color.shape[1]
        # print(f"{avg_movie_w_h = }")

        frame_num_to_epoch_name = {}
        for rng, epoch in zip(row.epochs_ranges, row.epochs_names):
            frame_num_to_epoch_name.update(
                              # idx+1: Frames start from 1
                              {idx+1:epoch  for idx in range(rng[0], rng[1]+1)})

        areas_df = self._info_df_dict[row.info_df_id]
        Xs_valid, Ys_valid, Xs_invalid, Ys_invalid, clrs = \
                                    self._extractTraces(areas_df, avg_movie_w_h,
                                                        self._only_traces_ids,
                                                        self._areas_to_colors)
        avg_movie_color, figsize = self._drawColoredBrainRegionOnMovie(
                                                          avg_movie_color,
                                                          avg_movie_w_h,
                                                          Xs_valid=Xs_valid,
                                                          Ys_valid=Ys_valid,
                                                          Xs_invalid=Xs_invalid,
                                                          Ys_invalid=Ys_invalid,
                                                          clrs=clrs)
        avg_mov_w_trace = self._renderPlotAndVid(ax_plt=ax_plt,
                                avg_movie_color=avg_movie_color, fps=fps,
                                frame_num_to_epoch_name=frame_num_to_epoch_name,
                                figsize=figsize)

        save_fp = self._savePrefix(plot_id=row.TrialNumber, df=data,
                                   session_name=row.ShortName)
        if self._save_tiff:
            save_fp_tiff = Path(str(save_fp)[:-4] + "overlaid.tiff")
            _saveTiff(avg_mov_w_trace, save_fp=save_fp_tiff)
        _saveMovie(avg_mov_w_trace, fps=fps, save_fp=save_fp)
        return data

    @staticmethod
    def _extractTraces(areas_df, avg_movie_w_h, only_traces_ids,
                       areas_to_colors):
        '''Get overlapping atlas'''
        Xs_valid, Ys_valid, Xs_invalid, Ys_invalid, Cs = [], [], [], [], []
        w, h = avg_movie_w_h
        # ex_area = areas_df.area_name.iloc[0]
        # has_dir = ex_area.endswith("_left") or ex_area.endswith("_right")
        for _, area_row in areas_df.iterrows():
            name = area_row.area_name
            name_stripped = name.rsplit("_left", 1)[0].rsplit("_right", 1)[0]
            if len(only_traces_ids) and name not in only_traces_ids and \
               name_stripped not in only_traces_ids:
                continue
            if callable(areas_to_colors):
                clr = areas_to_colors(area_row.acronym)
            elif name in areas_to_colors or \
                 name_stripped in areas_to_colors:
                clr = areas_to_colors[area_row.acronym]
            else:
                clr = 'k'
            valid_contour = area_row.area_valid_contour #area_valid_pix
            invalid_contour = area_row.area_invalid_pix # Use pixels
            # TODO: Use already computed contours
            for contour, Xs_list, Ys_list in [
                                     (valid_contour, Xs_valid,Ys_valid),
                                     (invalid_contour, Xs_invalid, Ys_invalid)]:
                if len(contour):
                    Xs = contour[:,0]
                    Xs[Xs >= w] = w - 1
                    Ys = contour[:,1]
                    Ys[Ys >= h] = h - 1
                else:
                    Xs, Ys = [], []
                Xs_list.append(Xs)
                Ys_list.append(Ys)
            Cs.append(clr)
        # print(f"{len(trials_traces_by_area_df.area.unique()) = }")
        # print(f"{len(areas_traces) = }")
        return Xs_valid, Ys_valid, Xs_invalid, Ys_invalid, Cs

    def _drawColoredBrainRegionOnMovie(self, avg_movie_color, avg_movie_w_h,
                                       Xs_valid, Ys_valid, Xs_invalid,
                                       Ys_invalid, clrs):
        fig = plt.gcf()
        fig_width, fig_height = fig.get_size_inches()
        plt.close(fig) # We will create a new figure everytime we need one
        mov_w, mov_h = avg_movie_w_h
        figsize = mov_w/self._dpi, mov_h/self._dpi

        fig = plt.figure(figsize=figsize, dpi=self._dpi)
        tmp_ax = fig.add_subplot()
        avg_movie_color2 = []
        tmp_ax.clear()
        ln = len(clrs)
        for zip_obj in [zip(Xs_valid,   Ys_valid,   clrs, [0.1]*ln),
                        zip(Xs_invalid, Ys_invalid, clrs, [0.5]*ln)]:
            for x, y, c, alpha in zip_obj:
                if not len(x):
                    continue
                tmp_ax.plot(y, x, c=c, linestyle='none', marker=".", ms=.5,
                            alpha=alpha)
                # tmp_ax.scatter(y, x, c=c, alpha=alpha*0.1, marker='s',
                #                s=self._dpi)
        org_lines = [line for line in tmp_ax.lines]
        for frame_color in tqdm(avg_movie_color, desc="Overlaying atlas"):
            tmp_ax.clear()
            tmp_ax.axis('off')
            tmp_ax.imshow(frame_color, origin="lower")
            # tmp_ax.lines += org_lines
            [tmp_ax.add_line(line) for line in org_lines]
            tmp_ax.set_aspect(mov_w/mov_h)
            plt.tight_layout(pad=0)
            fig.canvas.draw()
            # plt.show()
            # return
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Drop the alpha channel
            data = data[...,:3]
            # buf.seek(0)
            # format = "tiff"
            # plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
            # buf.seek(0)
            # data = plt.imread(buf, format=format)[:,:,:3]
            avg_movie_color2.append(data)
        avg_movie_color = np.array(avg_movie_color2)
        plt.close('all')
        return avg_movie_color, figsize

    def _renderPlotAndVid(self, ax_plt, avg_movie_color,
                          frame_num_to_epoch_name, fps, figsize):
        '''ax_plt should already be plotted. It be from in this method.'''
        org_data = [(line, line.get_xdata(), line.get_ydata())
                    for line in ax_plt.lines]
        max_x = max([xs[-1] for l, xs, ys in org_data])
        assert avg_movie_color.shape[0] == max_x, (
               f"{max_x = } != {avg_movie_color.shape = }")

        fig = plt.figure(figsize=figsize, dpi=self._dpi)
        plt.figure(fig.number)
        # print("new size: ", fig.get_size_inches())
        avg_movie_w_trace = []
        org_title = ax_plt.get_title()
        buf = io.BytesIO()
        for i, frame_color in enumerate(tqdm(avg_movie_color,
                                            desc="Placing plot next to movie")):
            # print("frame_color shape:", frame_color.shape)
            ax_plt = _redrawPlotUntilFrame(ax_plt, ax_all_lines=org_data,
                                           frame_num=i+1)
            # ax_plt.axis('off')
            if (i + 1) in frame_num_to_epoch_name:
                epoch_name = f"{frame_num_to_epoch_name[i+1]}"
            else:
                epoch_name = "*UNKNOWN*"
            ax_plt.set_title(f"{org_title} - {epoch_name} - Sec: {i/fps:.1f}",
                             fontsize="x-small")
            plt.sca(ax_plt)
            plt.draw()
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            buf.seek(0)
            format = "tiff"
            plt.savefig(buf, format=format, dpi=self._dpi, bbox_inches='tight')
            buf.seek(0)
            ax_img = plt.imread(buf, format=format)[:,:,:3]
            width_height = fig.canvas.get_width_height()[::-1]
            # print("Resizing: ", ax_img.shape, "to:", (width_height[1],
            # width_height[0]))
            ax_img = cv2.resize(ax_img, (width_height[1], width_height[0]),
                                interpolation=cv2.INTER_AREA)
            total_frame = np.append(frame_color, ax_img, axis=1)
            # print("total frame shape:", total_frame.shape, "dtype:",
            #             total_frame.dtype)
            avg_movie_w_trace.append(total_frame)
            # plt.show()
        plt.close('all')
        buf.close()
        avg_movie_w_trace = np.array(avg_movie_w_trace)
        print("avg_movie_color.shape:", avg_movie_w_trace.shape)
        return avg_movie_w_trace

    def _defaultSavePrefix(self, plot_id, session_name):
        save_prefix = f"{self._save_prefix}/{session_name}/{plot_id}.avi"
        return save_prefix

    def descr(self):
        return "Saving data of plotted traces along with movie"

def _saveTiff(mov, save_fp, traces_min=None, traces_max=None):
    print("Mov shape before:", mov.shape, mov.dtype)
    if True:
        mov = mov.astype(np.float64)
        mov_std = np.std(mov, axis=0)
        # We will set outside values, i.e no variation in fluorescence, to
        # white later. However, we want first to calculate the min-max values
        # of only the valid pixels. So initially, we set outside values to
        # nan so it'd be ignored when calculating the min-max percentiles next.
        mov[:, mov_std < 0.0001] = np.nan
        # Normalize the movie between almost min and almost max to ignore
        # few extreme values within the active values.
        min_in_t, max_in_t = np.nanpercentile(mov, [1, 99], axis=None)
        print(f"Full: {save_fp = } - {min_in_t = } - { max_in_t = }")
        mov[:, mov_std < 0.0001] = max_in_t # Set to white
        mov = np.clip(mov, min_in_t, max_in_t)
        # Normalize between 0 -> 1
        mov = (mov - min_in_t) / (max_in_t - min_in_t)
    else:
        mov = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        for frame in mov])
        # mov = (mov-mov.min()) / (mov.max() - mov.min())
        # min_in_t = mov.min(axis=0)
        # max_in_t = mov.max(axis=0)
        mov = mov.astype(float)
        mov_std = np.std(mov, axis=0)
        mov[:, mov_std < 0.0001] = np.nan
        min_in_t, max_in_t = np.nanpercentile(mov, [1, 99], axis=None)#(1, 2))
        mov[:, mov_std < 0.0001] = max_in_t # Set to white
        if save_fp.parent.stem == "WF3_M12":
            min_in_t, max_in_t = 49.0, 213.0
        elif save_fp.parent.stem == "WF3_M12":
            min_in_t, max_in_t = 48.0, 207.0
        print("min_in_t:",min_in_t , "max_in_t:", max_in_t)
        mov = (mov-min_in_t) / (max_in_t - min_in_t)
    # Select the color mao to use from here
    # colormap = plt.get_cmap('plasma')
    # colormap = None
    # colormap = plt.get_cmap('inferno')
    # colormap = colorMapParula()
    colormap = colorMapFireFiji()

    if colormap is not None:
        mov = colormap(mov)
        colormap_name = f"_{colormap.name}"
        if mov.ndim == 4:
            mov = mov[:,:,:,:3]
    else:
        colormap = ""
    mov_min, mov_max = mov.min(), mov.max()
    print("Mov min:", mov_min, "- Mov max:    ", mov_max)
    print("Mov shape:", mov.shape, mov.dtype)
    # Save the colormap:
    if (colormap is not None and traces_min is not None and
        traces_max is not None):
        print("Saving colormap")
        pts = colormap(np.linspace(0, 1, 500, endpoint=True))
        fig_cmap, cmap_ax = plt.subplots(1, 1, figsize=(8, 0.5))
        print("Pts.shape", pts.shape)
        pts = pts.reshape(1, -1, 4)
        print("Pts.shape", pts.shape)
        cmap_ax.imshow(pts,    aspect='auto')
        cmap_ax.yaxis.set_visible(False)
        cmap_ax.xaxis.set_ticks([0, pts.shape[1]])
        cmap_ax.xaxis.set_ticklabels([traces_min, traces_max])
        cmap_ax.spines['bottom'].set_visible(False)
        cmap_ax.spines['left'].set_visible(False)
        cmap_ax.tick_params(axis='x', which='both', bottom=False, top=False,
                            labelbottom=True)
        clr_map_fp = Path(save_fp.parent, f"{colormap_name[1:]}.svg")
        fig_cmap.savefig(clr_map_fp, bbox_inches='tight')
        plt.close(fig_cmap)
    save_fp = Path(save_fp.parent, f"{save_fp.stem}{colormap_name}.tiff")
    tifffile.imsave(save_fp, mov)

def _saveMovie(mov, fps, save_fp):
    save_fp = save_fp.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    write_params = [str(save_fp), fourcc, fps, (mov.shape[2], mov.shape[1])]
    # print("Vid shape:",(mov.shape[1], mov.shape[2]))
    labeled_video = cv2.VideoWriter(*write_params)
    for img in tqdm(mov):
        BGR_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        labeled_video.write(BGR_img)
    print("Closing avi file")
    labeled_video.release()


def _redrawPlotUntilFrame(ax, ax_all_lines : List[Line2D], frame_num):
    [line.remove() for line in ax.lines]
    for line, line_xdata, line_ydata in ax_all_lines:
        # If it's a label, don't do anything and add it as is
        if not (len(line_xdata) == 2 and (line_xdata[0] == line_xdata[1] or
                                          line_ydata[0] == line_ydata[1])):
            if line_xdata[0] > frame_num:
                continue
            max_idx = min(line_xdata[-1], frame_num) - line_xdata[0]
            line.set_xdata(line_xdata[:max_idx])
            line.set_ydata(line_ydata[:max_idx])
        # ax.lines.append(line)
        ax.add_line(line)
    # ax.lines.append(line)
    ax.add_line(line)
    return ax