// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "GSDReader.h"
#include "ExecutionConfiguration.h"
#include "GSD.h"
#include "SnapshotSystemData.h"
#include "hoomd/extern/gsd.h"
#include <sstream>
#include <string.h>

#include <stdexcept>
using namespace std;
using namespace hoomd::detail;

namespace hoomd
    {
/*! \param exec_conf The execution configuration
    \param name File name to read
    \param frame Frame index to read from the file
    \param from_end Count frames back from the end of the file

    The GSDReader constructor opens the GSD file, initializes an empty snapshot, and reads the file
   into memory (on the root rank).
*/
GSDReader::GSDReader(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                     const std::string& name,
                     const uint64_t frame,
                     bool from_end)
    : m_exec_conf(exec_conf), m_timestep(0), m_name(name), m_frame(frame)
    {
    m_snapshot = std::make_shared<SnapshotSystemData<double>>();

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (!m_exec_conf->isRoot())
        {
        return;
        }
#endif

    // open the GSD file in read mode
    m_exec_conf->msg->notice(3) << "data.gsd_snapshot: open gsd file " << name << endl;
    int retval = gsd_open(&m_handle, name.c_str(), GSD_OPEN_READONLY);
    GSDUtils::checkError(retval, m_name);

    // validate schema
    if (string(m_handle.header.schema) != string("hoomd"))
        {
        std::ostringstream s;
        s << "Invalid schema in " << name << endl;
        throw runtime_error(s.str());
        }
    if (m_handle.header.schema_version >= gsd_make_version(2, 1))
        {
        std::ostringstream s;
        s << "Invalid schema version in " << name << endl;
        throw runtime_error(s.str());
        }

    // set frame from the end of the file if requested
    uint64_t nframes = gsd_get_nframes(&m_handle);
    if (from_end && frame <= nframes)
        m_frame = nframes - frame;

    // validate number of frames
    if (m_frame >= nframes)
        {
        std::ostringstream s;
        s << "Cannot read frame " << m_frame << " " << name << " only has "
          << gsd_get_nframes(&m_handle) << " frames.";
        throw runtime_error(s.str());
        }

    readHeader();
    readParticles();
    readTopology();
    }

GSDReader::~GSDReader()
    {
#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (!m_exec_conf->isRoot())
        {
        return;
        }
#endif

    gsd_close(&m_handle);
    }

/*! \param data Pointer to data to read into
    \param frame Frame index to read from
    \param name Name of the data chunk
    \param expected_size Expected size of the data chunk in bytes.
    \param cur_n N in the current frame.

    Attempts to read the data chunk of the given name at the given frame. If it is not present at
   this frame, attempt to read from frame 0. If it is also not present at frame 0, return false. If
   the found data chunk is not the expected size, throw an exception.

    Per the GSD spec, keep the default when the frame 0 N does not match the current N.

    Return true if data is actually read from the file.
*/
bool GSDReader::readChunk(void* data,
                          uint64_t frame,
                          const char* name,
                          size_t expected_size,
                          unsigned int cur_n)
    {
    const struct gsd_index_entry* entry = gsd_find_chunk(&m_handle, frame, name);
    if (entry == NULL && frame != 0)
        entry = gsd_find_chunk(&m_handle, 0, name);

    if (entry == NULL || (cur_n != 0 && entry->N != cur_n))
        {
        m_exec_conf->msg->notice(10) << "data.gsd_snapshot: chunk not found " << name << endl;
        return false;
        }
    else
        {
        m_exec_conf->msg->notice(7) << "data.gsd_snapshot: reading chunk " << name << endl;
        size_t actual_size = entry->N * entry->M * gsd_sizeof_type((enum gsd_type)entry->type);

        // Check if we need to convert from float to double
        if (entry->type == GSD_TYPE_FLOAT && expected_size == actual_size * 2)
            {
            // convert floats to double
            size_t num_elements = entry->N * entry->M;
            std::vector<float> float_buffer(num_elements);
            int retval = gsd_read_chunk(&m_handle, float_buffer.data(), entry);
            GSDUtils::checkError(retval, m_name);
            
            // Cast float to double
            double* double_data = reinterpret_cast<double*>(data);
            for (size_t i = 0; i < num_elements; i++)
                {
                double_data[i] = static_cast<double>(float_buffer[i]);
                }
            }
        else {
            if (actual_size != expected_size)
                {
                std::ostringstream s;
                s << "Expecting " << expected_size << " bytes in " << name << " but found "
                << actual_size << ".";
                throw runtime_error(s.str());
                }
            int retval = gsd_read_chunk(&m_handle, data, entry);
            GSDUtils::checkError(retval, m_name);
        }

        return true;
        }
    }

/*! \param frame Frame index to read from
    \param name Name of the data chunk

    Attempts to read the data chunk of the given name at the given frame. If it is not present at
   this frame, attempt to read from frame 0. If it is also not present at frame 0, return an empty
   list.

    If the data chunk is found in the file, return a vector of string type names.
*/
std::vector<std::string> GSDReader::readTypes(uint64_t frame, const char* name)
    {
    m_exec_conf->msg->notice(7) << "data.gsd_snapshot: reading chunk " << name << endl;

    std::vector<std::string> type_mapping;

    // set the default particle type mapping per the GSD HOOMD Schema
    if (std::string(name) == "particles/types")
        type_mapping.push_back("A");

    const struct gsd_index_entry* entry = gsd_find_chunk(&m_handle, frame, name);
    if (entry == NULL && frame != 0)
        entry = gsd_find_chunk(&m_handle, 0, name);

    if (entry == NULL)
        return type_mapping;
    else
        {
        size_t actual_size = entry->N * entry->M * gsd_sizeof_type((enum gsd_type)entry->type);
        std::vector<char> data(actual_size);
        int retval = gsd_read_chunk(&m_handle, &data[0], entry);
        GSDUtils::checkError(retval, m_name);

        type_mapping.clear();
        for (unsigned int i = 0; i < entry->N; i++)
            {
            size_t l = strnlen(&data[i * entry->M], entry->M);
            type_mapping.push_back(std::string(&data[i * entry->M], l));
            }

        return type_mapping;
        }
    }

/*! Read the same data chunks written by GSDDumpWriter::writeFrameHeader
 */
void GSDReader::readHeader()
    {
    readChunk(&m_timestep, m_frame, "configuration/step", 8);

    uint8_t dim = 3;
    readChunk(&dim, m_frame, "configuration/dimensions", 1);

    float box[6] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    readChunk(&box, m_frame, "configuration/box", 6 * 4);
    auto snap = std::static_pointer_cast<SnapshotSystemData<double>>(m_snapshot);
    snap->dimensions = dim;
    
    if (dim == 2)
        {
        box[2] = 0;
        box[4] = 0;
        box[5] = 0;
        }
    snap->global_box = std::make_shared<BoxDim>(BoxDim(box[0], box[1], box[2]));
    snap->global_box->setTiltFactors(box[3], box[4], box[5]);

    unsigned int N = 0;
    readChunk(&N, m_frame, "particles/N", 4);
    if (N == 0)
        {
        std::ostringstream s;
        s << "Cannot read a file with 0 particles.";
        throw runtime_error(s.str());
        }
    snap->particle_data.resize(N);
    }

/*! Read the same data chunks for particles
 */
void GSDReader::readParticles()
    {
    auto snap = std::static_pointer_cast<SnapshotSystemData<double>>(m_snapshot);
    unsigned int N = snap->particle_data.size;
    snap->particle_data.type_mapping = readTypes(m_frame, "particles/types");
    
    size_t scalar_bytes = 8;  // double
    size_t vec3_bytes = 24;   // 3 * 8
    size_t quat_bytes = 32;   // 4 * 8
    
    readChunk(snap->particle_data.charge.data(), m_frame, "particles/charge", N * scalar_bytes, N);
    readChunk(snap->particle_data.type.data(), m_frame, "particles/typeid", N * 4, N);
    readChunk(snap->particle_data.mass.data(), m_frame, "particles/mass", N * scalar_bytes, N);
    readChunk(snap->particle_data.diameter.data(), m_frame, "particles/diameter", N * scalar_bytes, N);
    readChunk(snap->particle_data.body.data(), m_frame, "particles/body", N * 4, N);
    readChunk(snap->particle_data.inertia.data(), m_frame, "particles/moment_inertia", N * vec3_bytes, N);
    readChunk(snap->particle_data.pos.data(), m_frame, "particles/position", N * vec3_bytes, N);
    readChunk(snap->particle_data.orientation.data(), m_frame, "particles/orientation", N * quat_bytes, N);
    readChunk(snap->particle_data.vel.data(), m_frame, "particles/velocity", N * vec3_bytes, N);
    readChunk(snap->particle_data.angmom.data(), m_frame, "particles/angmom", N * quat_bytes, N);
    readChunk(snap->particle_data.image.data(), m_frame, "particles/image", N * 12, N);
    }

/*! Read the same data chunks for topology
 */
void GSDReader::readTopology()
    {
    auto snap = std::static_pointer_cast<SnapshotSystemData<double>>(m_snapshot);
    unsigned int N = 0;
    snap->bond_data.type_mapping = readTypes(m_frame, "bonds/types");
    readChunk(&N, m_frame, "bonds/N", 4);
    if (N > 0)
        {
        snap->bond_data.resize(N);
        readChunk(snap->bond_data.type_id.data(), m_frame, "bonds/typeid", N * 4, N);
        readChunk(snap->bond_data.groups.data(), m_frame, "bonds/group", N * 8, N);
        }

    N = 0;
    snap->angle_data.type_mapping = readTypes(m_frame, "angles/types");
    readChunk(&N, m_frame, "angles/N", 4);
    if (N > 0)
        {
        snap->angle_data.resize(N);
        readChunk(snap->angle_data.type_id.data(), m_frame, "angles/typeid", N * 4, N);
        readChunk(snap->angle_data.groups.data(), m_frame, "angles/group", N * 12, N);
        }

    N = 0;
    snap->dihedral_data.type_mapping = readTypes(m_frame, "dihedrals/types");
    readChunk(&N, m_frame, "dihedrals/N", 4);
    if (N > 0)
        {
        snap->dihedral_data.resize(N);
        readChunk(snap->dihedral_data.type_id.data(), m_frame, "dihedrals/typeid", N * 4, N);
        readChunk(snap->dihedral_data.groups.data(), m_frame, "dihedrals/group", N * 16, N);
        }

    N = 0;
    snap->improper_data.type_mapping = readTypes(m_frame, "impropers/types");
    readChunk(&N, m_frame, "impropers/N", 4);
    if (N > 0)
        {
        snap->improper_data.resize(N);
        readChunk(snap->improper_data.type_id.data(), m_frame, "impropers/typeid", N * 4, N);
        readChunk(snap->improper_data.groups.data(), m_frame, "impropers/group", N * 16, N);
        }

    N = 0;
    readChunk(&N, m_frame, "constraints/N", 4);
    if (N > 0)
        {
        snap->constraint_data.resize(N);
        std::vector<float> data(N);
        readChunk(&data[0], m_frame, "constraints/value", N * 4, N);
        for (unsigned int i = 0; i < N; i++)
            snap->constraint_data.val[i] = Scalar(data[i]);
        readChunk(snap->constraint_data.groups.data(), m_frame, "constraints/group", N * 8, N);
        }

    if (m_handle.header.schema_version >= gsd_make_version(1, 1))
        {
        N = 0;
        snap->pair_data.type_mapping = readTypes(m_frame, "pairs/types");
        readChunk(&N, m_frame, "pairs/N", 4);
        if (N > 0)
            {
            snap->pair_data.resize(N);
            readChunk(snap->pair_data.type_id.data(), m_frame, "pairs/typeid", N * 4, N);
            readChunk(snap->pair_data.groups.data(), m_frame, "pairs/group", N * 8, N);
            }
        }
    }

pybind11::list GSDReader::readTypeShapesPy(uint64_t frame)
    {
    std::vector<std::string> type_mapping = this->readTypes(frame, "particles/type_shapes");
    pybind11::list type_shapes;
    for (unsigned int i = 0; i < type_mapping.size(); i++)
        type_shapes.append(type_mapping[i]);
    return type_shapes;
    }

namespace detail
    {
void export_GSDReader(pybind11::module& m)
    {
    pybind11::class_<GSDReader, std::shared_ptr<GSDReader>>(m, "GSDReader")
        .def(pybind11::init<std::shared_ptr<const ExecutionConfiguration>,
                            const string&,
                            const uint64_t,
                            bool>())
        .def("getTimeStep", &GSDReader::getTimeStep)
        .def("getSnapshot", &GSDReader::getSnapshot)
        .def("clearSnapshot", &GSDReader::clearSnapshot)
        .def("readTypeShapesPy", &GSDReader::readTypeShapesPy);
    }

    } // end namespace detail

    } // end namespace hoomd
