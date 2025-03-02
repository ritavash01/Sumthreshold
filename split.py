import numpy as np 
from sigpyproc.readers import FilReader
import matplotlib.pyplot as plt
import mmap
from pathlib import Path
from dataclasses import dataclass
from priwo import readhdr, writefil
import os 

@dataclass
class SIGPROCFilterbank:
    """
    Context manager to read data from a SIGPROC filterbank file.
    """

    filepath: str | Path

    def __enter__(self):
        self.header = readhdr(str(self.filepath))

        try:
            self.fh = self.header["fch1"]
            self.df = self.header["foff"]
            self.dt = self.header["tsamp"]
            self.nf = self.header["nchans"]
            self.nskip = self.header["size"]
            self.nbits = self.header["nbits"]
        except KeyError:
            print("File may not be a valid SIGPROC filterbank file.")

        self.fobject = open(self.filepath, mode="rb")

        if self.df < 0:
            self.df = abs(self.df)
            self.bw = self.nf * self.df
            self.fl = self.fh - self.bw + (0.5 * self.df)
        else:
            self.fl = self.fh
            self.bw = self.nf * self.df
            self.fh = self.fl + self.bw - (0.5 * self.df)

        self.dtype = {
            8: np.uint8,
            16: np.uint16,
            32: np.float32,
            64: np.float64,
        }[self.nbits]

        self.mapped = mmap.mmap(
            self.fobject.fileno(),
            0,
            mmap.MAP_PRIVATE,
            mmap.PROT_READ,
        )

        self.nt = int(int(self.mapped.size() - self.nskip) / self.nf)

        return self

    def __exit__(self, *args) -> None:
        if self.fobject:
            self.fobject.close()

    def get(self, offset: int, count: int) -> np.ndarray:
        """
        Get data from this file.
        """

        data = np.frombuffer(
            self.mapped,
            dtype=np.uint8,
            count=count * self.nf,
            offset=offset * self.nf + self.nskip,
        )
        data = data.reshape(-1, self.nf)
        data = data.T
        return data



# Paths
input_filepath = "/home/ritavash/Documents/filterbank_data/Run2_Dynamic_FRB_SHM_B0329_B4_800BMs_50BMPH.raw.fil"
output_dir = "/mnt/disk1/data/Ritavash/RFI_test/july17/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read data from the filterbank file
with SIGPROCFilterbank(input_filepath) as fb:
    chunk_size = 128000  # Number of time samples per chunk
    total_samples =   # Total samples to process
    
    num_chunks = total_samples // chunk_size  # Number of full chunks
    remainder = total_samples % chunk_size  # Remaining samples (if any)

    for i in range(num_chunks):
        data = fb.get(offset=i * chunk_size, count=chunk_size).astype(np.int8)
        filename = os.path.join(output_dir, f"chunk_{i}.dat")
        data.tofile(filename)  # Save as binary int8 format
        print(f"Saved {filename}")

    # Handle remainder samples (if any)
    if remainder > 0:
        data = fb.get(offset=num_chunks * chunk_size, count=remainder).astype(np.int8)
        filename = os.path.join(output_dir, f"chunk_{num_chunks}.dat")
        data.tofile(filename)
        print(f"Saved {filename}")
