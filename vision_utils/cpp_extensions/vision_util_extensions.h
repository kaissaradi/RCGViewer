#include <string>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py=pybind11;

py::bytes pack_sta_buffer(py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
    py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
    double stixel_size) {
    
    /* Pack a single STA into a byte buffer: sta[t,x,y,color] */
    py::buffer_info red_sta_info = red_sta.request();
    float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

    size_t sta_depth = red_sta_info.shape[0];
    size_t sta_width = red_sta_info.shape[1];
    size_t sta_height = red_sta_info.shape[2];

    size_t n_output_entries = 6 * sta_width * sta_height * sta_depth + sta_depth * 4;

    py::buffer_info red_err_info = red_err.request();
    float *red_err_ptr = static_cast<float *> (red_err_info.ptr);

    py::buffer_info green_sta_info = green_sta.request();
    float *green_data_ptr = static_cast<float *> (green_sta_info.ptr);

    py::buffer_info green_err_info = green_err.request();
    float *green_err_ptr = static_cast<float *> (green_err_info.ptr);

    py::buffer_info blue_sta_info = blue_sta.request();
    float *blue_data_ptr = static_cast<float *> (blue_sta_info.ptr);

    py::buffer_info blue_err_info = blue_err.request();
    float *blue_err_ptr = static_cast<float *> (blue_err_info.ptr);

    uint32_t *output_buffer = new uint32_t[n_output_entries];

    size_t depth_offset, width_depth_offset, read_offset;
    uint64_t stixel_temp;
    size_t write_idx = 0;
    for (size_t i = 0; i < sta_depth; i++) {

        depth_offset = i * (sta_width * sta_height);

        output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_height));
        output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_width));

        stixel_temp = __builtin_bswap64(*(reinterpret_cast<uint64_t *>(&stixel_size)));
        output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp >> 32);
        output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp & 0xFFFF);

        for (size_t j = 0; j < sta_width; j++)  {

            width_depth_offset = j * sta_height + depth_offset;

            for (size_t k = 0; k < sta_height; k++) {

                read_offset = width_depth_offset + k;

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_err_ptr + read_offset)));

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_err_ptr + read_offset)));

                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_data_ptr + read_offset)));
                output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_err_ptr + read_offset)));
            }
        }
    }
    return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
}

// py::bytes pack_sta_buffer(py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
//     double stixel_size) {
    
//     /* Pack a single STA into a byte buffer: sta[t,x,y,color] */
//     py::buffer_info red_sta_info = red_sta.request();
//     float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

//     size_t sta_width = red_sta_info.shape[0];
//     size_t sta_height = red_sta_info.shape[1];

//     uint32_t s_width = 95;
//     uint32_t s_height = 152;

//     // py::print("sta_width: ", sta_width);
//     // py::print("sta_height: ", sta_height);

//     // size_t n_output_entries = 6 * sta_width * sta_height + 2;
//     size_t n_output_entries = 6 * sta_width * sta_height + 4;

//     py::buffer_info red_err_info = red_err.request();
//     float *red_err_ptr = static_cast<float *> (red_err_info.ptr);

//     py::buffer_info green_sta_info = green_sta.request();
//     float *green_data_ptr = static_cast<float *> (green_sta_info.ptr);

//     py::buffer_info green_err_info = green_err.request();
//     float *green_err_ptr = static_cast<float *> (green_err_info.ptr);

//     py::buffer_info blue_sta_info = blue_sta.request();
//     float *blue_data_ptr = static_cast<float *> (blue_sta_info.ptr);

//     py::buffer_info blue_err_info = blue_err.request();
//     float *blue_err_ptr = static_cast<float *> (blue_err_info.ptr);

//     uint32_t *output_buffer = new uint32_t[n_output_entries];

//     size_t width_depth_offset, read_offset;
//     uint64_t stixel_temp;
//     size_t write_idx = 0;

//     // output_buffer[write_idx++] = __builtin_bswap32(s_height); //__builtin_bswap32(s_width);
//     // output_buffer[write_idx++] = __builtin_bswap32(s_width); //__builtin_bswap32(s_height);

//     // output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_width));
//     // output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_height));
//     output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_height));
//     output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_width));

//     stixel_temp = __builtin_bswap64(*(reinterpret_cast<uint64_t *>(&stixel_size)));
//     output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp >> 32);
//     output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp & 0xFFFF);

//     for (size_t j = 0; j < sta_width; j++)  {

//         width_depth_offset = j * sta_height;

//         for (size_t k = 0; k < sta_height; k++) {

//             read_offset = width_depth_offset + k;

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_err_ptr + read_offset)));

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_err_ptr + read_offset)));

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_err_ptr + read_offset)));
//         }
//     }


//     return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
// }

// py::bytes pack_sta_buffer(py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
//     double stixel_size) {
    
//     /* Pack a single STA into a byte buffer: sta[t,x,y,color] */
//     py::buffer_info red_sta_info = red_sta.request();
//     float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

//     size_t sta_width = red_sta_info.shape[0];
//     size_t sta_height = red_sta_info.shape[1];

//     size_t n_output_entries = 6 * sta_width * sta_height;

//     py::buffer_info red_err_info = red_err.request();
//     float *red_err_ptr = static_cast<float *> (red_err_info.ptr);

//     py::buffer_info green_sta_info = green_sta.request();
//     float *green_data_ptr = static_cast<float *> (green_sta_info.ptr);

//     py::buffer_info green_err_info = green_err.request();
//     float *green_err_ptr = static_cast<float *> (green_err_info.ptr);

//     py::buffer_info blue_sta_info = blue_sta.request();
//     float *blue_data_ptr = static_cast<float *> (blue_sta_info.ptr);

//     py::buffer_info blue_err_info = blue_err.request();
//     float *blue_err_ptr = static_cast<float *> (blue_err_info.ptr);

//     uint32_t *output_buffer = new uint32_t[n_output_entries];

//     size_t width_depth_offset, read_offset;
//     size_t write_idx = 0;

//     for (size_t j = 0; j < sta_width; j++)  {

//         width_depth_offset = j * sta_height;

//         for (size_t k = 0; k < sta_height; k++) {

//             read_offset = width_depth_offset + k;

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (red_err_ptr + read_offset)));

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (green_err_ptr + read_offset)));

//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_data_ptr + read_offset)));
//             output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (blue_err_ptr + read_offset)));
//         }
//     }


//     return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
// }


// py::bytes pack_sta_buffer(py::array_t<float, py::array::c_style | py::array::forcecast>& red_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& red_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& green_err,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_sta,
//     py::array_t<float, py::array::c_style | py::array::forcecast>& blue_err,
//     double stixel_size) {
    
//     /* Pack a single STA into a byte buffer: sta[t,x,y,color] */
//     py::buffer_info red_sta_info = red_sta.request();
//     float *red_data_ptr = static_cast<float *> (red_sta_info.ptr);

//     size_t sta_depth = red_sta_info.shape[0];
//     size_t sta_width = red_sta_info.shape[1];
//     size_t sta_height = red_sta_info.shape[2];

//     size_t n_output_entries = 6 * sta_width * sta_height * sta_depth + sta_depth * 4;

//     uint32_t *output_buffer = new uint32_t[n_output_entries];

//     size_t depth_offset, width_depth_offset, read_offset;
//     uint64_t stixel_temp;
//     size_t write_idx = 0;
//     for (size_t i = 0; i < sta_depth; i++) {

//         depth_offset = i * (sta_width * sta_height);

//         output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_width));
//         output_buffer[write_idx++] = __builtin_bswap32(static_cast<uint32_t>(sta_height));

//         stixel_temp = __builtin_bswap64(*(reinterpret_cast<uint64_t *>(&stixel_size)));
//         output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp >> 32);
//         output_buffer[write_idx++] = static_cast<uint32_t> (stixel_temp & 0xFFFF);

//         for (size_t j = 0; j < sta_width; j++)  {

//             width_depth_offset = j * sta_height + depth_offset;

//             for (size_t k = 0; k < sta_height; k++) {

//                 read_offset = width_depth_offset + k;

//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&red_sta.at(i, j, k))));
//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&red_err.at(i, j, k))));

//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&green_sta.at(i, j, k))));
//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&green_err.at(i, j, k))));

//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&blue_sta.at(i, j, k))));
//                 output_buffer[write_idx++] = __builtin_bswap32(*(reinterpret_cast<uint32_t *> (&blue_err.at(i, j, k))));
//             }
//         }
//     }

//     return py::bytes(reinterpret_cast<char *> (output_buffer), n_output_entries * 4);
// }
