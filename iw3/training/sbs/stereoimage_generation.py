# file from https://github.com/thygate/stable-diffusion-webui-depthmap-script
# MIT License / Copyright (c) 2023 Bob Thiry
from numba import njit, prange
import numpy as np
from PIL import Image

def create_stereoimages(original_image, depthmap, divergence, modes=None, stereo_balance=0.0,
                        fill_technique='polylines_sharp', convergence=1):
    """Creates stereoscopic images.
    An effort is made to make them look nice, but beware that the resulting image will have some distortion .

    :param original_image: original image from which the 3D image (stereoimage) will be created
    :param depthmap: depthmap corresponding to the original image. White = near, black = far.
    :param float divergence: the measure of 3D effect, in percentages.
      A good value will likely be somewhere in the [0.05; 10.0) interval.
    :param list modes: how the result will look like. By default only 'left-right' is generated
      - a picture for the left eye will be on the left and the picture from the right eye - on the rigth.
      The supported modes are: 'left-right', 'right-left', 'top-bottom', 'bottom-top', 'red-cyan-anaglyph'.
    :param float stereo_balance: has to do with how the divergence will be split among the two parts of the image,
      must be in the [-1.0; 1.0] interval.
    :param str fill_technique: applying divergence inevidably creates some gaps in the image.
      This parameter specifies the technique that will be used to fill in the blanks in the two resulting images.
      Must be one of the following: 'none', 'naive', 'naive_interpolating', 'polylines_soft', 'polylines_sharp'.
    """
    if modes is None:
        modes = ['left-right']
    if not isinstance(modes, list):
        modes = [modes]

    original_image = np.asarray(original_image)
    balance = (stereo_balance + 1) / 2
    left_eye = original_image if balance < 0.001 else \
        apply_stereo_divergence(original_image, depthmap, +1 * divergence * balance, fill_technique, convergence=convergence)
    right_eye = original_image if balance > 0.999 else \
        apply_stereo_divergence(original_image, depthmap, -1 * divergence * (1 - balance), fill_technique, convergence=convergence)

    results = []
    for mode in modes:
        if mode == 'left-right':
            results.append(np.hstack([left_eye, right_eye]))
        elif mode == 'right-left':
            results.append(np.hstack([right_eye, left_eye]))
        elif mode == 'top-bottom':
            results.append(np.vstack([left_eye, right_eye]))
        elif mode == 'bottom-top':
            results.append(np.vstack([right_eye, left_eye]))
        elif mode == 'red-cyan-anaglyph':
            results.append(overlap_red_cyan(left_eye, right_eye))
        else:
            raise Exception('Unknown mode')
    return [Image.fromarray(r) for r in results]


def apply_stereo_divergence(original_image, depth, divergence, fill_technique, convergence=1):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    divergence_px = (divergence / 100.0) * original_image.shape[1]

    if fill_technique in ['none', 'naive', 'naive_interpolating']:
        return apply_stereo_divergence_naive(original_image, normalized_depth, divergence_px, fill_technique, convergence=convergence)
    if fill_technique in ['polylines_soft', 'polylines_sharp']:
        return apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px, fill_technique, convergence=convergence)


@njit
def apply_stereo_divergence_naive(original_image, normalized_depth, divergence_px: float, fill_technique, convergence=1):
    h, w, c = original_image.shape

    derived_image = np.zeros_like(original_image)
    filled = np.zeros(h * w, dtype=np.uint8)

    for row in prange(h):
        # Swipe order should ensure that pixels that are closer overwrite
        # (at their destination) pixels that are less close
        for col in range(w) if divergence_px < 0 else range(w - 1, -1, -1):
            col_d = col + int((normalized_depth[row][col] ** 2) * divergence_px - divergence_px * convergence)
            if 0 <= col_d < w:
                derived_image[row][col_d] = original_image[row][col]
                filled[row * w + col_d] = 1

    # Fill the gaps
    if fill_technique == 'naive_interpolating':
        for row in range(h):
            for l_pointer in range(w):
                # This if (and the next if) performs two checks that are almost the same - for performance reasons
                if sum(derived_image[row][l_pointer]) != 0 or filled[row * w + l_pointer]:
                    continue
                l_border = derived_image[row][l_pointer - 1] if l_pointer > 0 else np.zeros(3, dtype=np.uint8)
                r_border = np.zeros(3, dtype=np.uint8)
                r_pointer = l_pointer + 1
                while r_pointer < w:
                    if sum(derived_image[row][r_pointer]) != 0 and filled[row * w + r_pointer]:
                        r_border = derived_image[row][r_pointer]
                        break
                    r_pointer += 1
                if sum(l_border) == 0:
                    l_border = r_border
                elif sum(r_border) == 0:
                    r_border = l_border
                # Example illustrating positions of pointers at this point in code:
                # is filled?  : +   -   -   -   -   +
                # pointers    :     l               r
                # interpolated: 0   1   2   3   4   5
                # In total: 5 steps between two filled pixels
                total_steps = 1 + r_pointer - l_pointer
                step = (r_border.astype(np.float_) - l_border) / total_steps
                for col in range(l_pointer, r_pointer):
                    derived_image[row][col] = l_border + (step * (col - l_pointer + 1)).astype(np.uint8)
        return derived_image
    elif fill_technique == 'naive':
        derived_fix = np.copy(derived_image)
        for pos in np.where(filled == 0)[0]:
            row = pos // w
            col = pos % w
            row_times_w = row * w
            for offset in range(1, abs(int(divergence_px)) + 2):
                r_offset = col + offset
                l_offset = col - offset
                if r_offset < w and filled[row_times_w + r_offset]:
                    derived_fix[row][col] = derived_image[row][r_offset]
                    break
                if 0 <= l_offset and filled[row_times_w + l_offset]:
                    derived_fix[row][col] = derived_image[row][l_offset]
                    break
        return derived_fix
    else:  # none
        return derived_image


@njit(parallel=True)  # fastmath=True does not reasonably improve performance
def apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px: float, fill_technique, convergence=1):
    # This code treats rows of the image as polylines
    # It generates polylines, morphs them (applies divergence) to them, and then rasterizes them
    EPSILON = 1e-7
    PIXEL_HALF_WIDTH = 0.45 if fill_technique == 'polylines_sharp' else 0.0
    # PERF_COUNTERS = [0, 0, 0]

    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    for row in prange(h):
        # generating the vertices of the morphed polyline
        # format: new coordinate of the vertex, divergence (closeness), column of pixel that contains the point's color
        pt = np.zeros((5 + 2 * w, 3), dtype=np.float_)
        pt_end: int = 0
        pt[pt_end] = [-3.0 * abs(divergence_px), 0.0, 0.0]
        pt_end += 1
        for col in range(0, w):
            coord_d = (normalized_depth[row][col] ** 2) * divergence_px - convergence * divergence_px
            coord_x = col + 0.5 + coord_d
            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2
        pt[pt_end] = [w + 3.0 * abs(divergence_px), 0.0, w - 1]
        pt_end += 1

        # generating the segments of the morphed polyline
        # format: coord_x, coord_d, color_i of the first point, then the same for the second point
        sg_end: int = pt_end - 1
        sg = np.zeros((sg_end, 6), dtype=np.float_)
        for i in range(sg_end):
            sg[i] += np.concatenate((pt[i], pt[i + 1]))
        # Here is an informal proof that this (morphed) polyline does not self-intersect:
        # Draw a plot with two axes: coord_x and coord_d. Now draw the original line - it will be positioned at the
        # bottom of the graph (that is, for every point coord_d == 0). Now draw the morphed line using the vertices of
        # the original polyline. Observe that for each vertex in the new polyline, its increments
        # (from the corresponding vertex in the old polyline) over coord_x and coord_d are in direct proportion.
        # In fact, this proportion is equal for all the vertices and it is equal either -1 or +1,
        # depending on the sign of divergence_px. Now draw the lines from each old vertex to a corresponding new vertex.
        # Since the proportions are equal, these lines have the same angle with an axe and are parallel.
        # So, these lines do not intersect. Now rotate the plot by 45 or -45 degrees and observe that
        # each dot of the polyline is further right from the last dot,
        # which makes it impossible for the polyline to self-interset. QED.

        # sort segments and points using insertion sort
        # has a very good performance in practice, since these are almost sorted to begin with
        for i in range(1, sg_end):
            u = i - 1
            while pt[u][0] > pt[u + 1][0] and 0 <= u:
                pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                u -= 1

        # rasterizing
        # at each point in time we keep track of segments that are "active" (or "current")
        csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float_)
        csg_end: int = 0
        sg_pointer: int = 0
        # and index of the point that should be processed next
        pt_i: int = 0
        for col in range(w):  # iterate over regions (that will be rasterizeed into pixels)
            color = np.full(c, 0.5, dtype=np.float_)  # we start with 0.5 because of how floats are converted to ints
            while pt[pt_i][0] < col:
                pt_i += 1
            pt_i -= 1  # pt_i now points to the dot before the region start
            # Finding segment' parts that contribute color to the region
            while pt[pt_i][0] < col + 1:
                coord_from = max(col, pt[pt_i][0]) + EPSILON
                coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
                significance = coord_to - coord_from
                # the color at center point is the same as the average of color of segment part
                coord_center = coord_from + 0.5 * significance

                # adding semgents that now may contribute
                while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
                    csg[csg_end] = sg[sg_pointer]
                    sg_pointer += 1
                    csg_end += 1
                # removing segments that will no longer contribute
                csg_i = 0
                while csg_i < csg_end:
                    if csg[csg_i][3] < coord_center:
                        csg[csg_i] = csg[csg_end - 1]
                        csg_end -= 1
                    else:
                        csg_i += 1
                # finding the closest segment (segment with most divergence)
                # note that this segment will be the closest from coord_from right up to coord_to, since there
                # no new segments "appearing" inbetween these two and _the polyline does not self-intersect_
                best_csg_i: int = 0
                # PERF_COUNTERS[0] += 1
                if csg_end != 1:
                    # PERF_COUNTERS[1] += 1
                    best_csg_closeness: float = -EPSILON
                    for csg_i in range(csg_end):
                        ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
                        # assert 0.0 <= ip_k <= 1.0
                        closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
                        if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
                            best_csg_closeness = closeness
                            best_csg_i = csg_i
                # getting the color
                col_l: int = int(csg[best_csg_i][2] + EPSILON)
                col_r: int = int(csg[best_csg_i][5] + EPSILON)
                if col_l == col_r:
                    color += original_image[row][col_l] * significance
                else:
                    # PERF_COUNTERS[2] += 1
                    ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
                    color += (original_image[row][col_l] * (1.0 - ip_k) +
                              original_image[row][col_r] * ip_k
                              ) * significance
                pt_i += 1
            derived_image[row][col] = np.asarray(color, dtype=np.uint8)
    # print(PERF_COUNTERS)
    return derived_image


@njit(parallel=True)
def overlap_red_cyan(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]

    # final image
    composite = np.zeros((height2, width2, 3), np.uint8)

    # iterate through "left" image, filling in red values of final image
    for i in prange(height1):
        for j in range(width1):
            composite[i, j, 0] = im1[i, j, 0]

    # iterate through "right" image, filling in blue/green values of final image
    for i in prange(height2):
        for j in range(width2):
            composite[i, j, 1] = im2[i, j, 1]
            composite[i, j, 2] = im2[i, j, 2]

    return composite
