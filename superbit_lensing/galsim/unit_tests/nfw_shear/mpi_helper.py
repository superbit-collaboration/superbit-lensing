# Helper that wraps mpi4py and allows fallback to serial
comm_except_list = []

def mpi_abort_excepthook(comm):
    """
    Override sys.excepthook to call MPI abort. Prevents stalled MPI operations
    when one process has an exception but others do not.
    """
    if comm in comm_except_list:
        return
    import sys
    syshook = sys.excepthook
    def new_excepthook(type, value, traceback):
        syshook(type, value, traceback)
        comm.Abort(1)
    sys.excepthook = new_excepthook
    comm_except_list.append(comm)

class MPIHelper:
    """
    Wrap MPI operations to provide serial fallback.

    If you ever want to efficiently communicate large numpy arrays, bug Steve
    to copy in those functions. Existing ones use more time/memory than needed,
    though they will still work.
    """

    def __init__(self, mpi=True,  mpi_root=0, comm=None):
        """
        Arguments
        ---------
        mpi : bool
            Whether to try to use MPI.
        mpi_root : int
            The process rank to use as root
        comm : mpi4py communicator
            If you want to specify one. By default uses COMM_WORLD
        """

        if mpi:
            try:
                from mpi4py import MPI
            except ImportError:
                mpi = False
        self.mpi = mpi

        if mpi:
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.mpi_rank = self.comm.Get_rank()
            self.mpi_size = self.comm.Get_size()
            self.rank_fmt = "MPI#{{:0{}d}} ".format(len(str(self.mpi_size-1)))
            self.rank_fmt = self.rank_fmt.format(self.mpi_rank)
            mpi_abort_excepthook(self.comm)
        else:
            self.comm = None
            self.mpi_rank = 0
            self.mpi_size = 1
            self.rank_fmt = ""

        self.mpi_root = mpi_root

        self.log("DEBUG {} MPI processes".format(self.mpi_size), root=True)

    def is_mpi_root(self):
        """
        Return True if this thread is the root MPI thread, otherwise False.
        """
        return (self.mpi_rank == self.mpi_root) or not self.mpi

    # TODO tie in with logging module
    def log(self, msg, root=False):
        """
        Log a message

        Arguments
        =========
        root : bool
            Log from MPI root process only
        """
        if not root or self.is_mpi_root():
            print(self.rank_fmt + msg)

    def barrier(self):
        """
        MPI barrier. Pause until all processes reach here.
        """
        if not self.mpi:
            return
        self.comm.Barrier()

    def none_except_root(self, data):
        """
        Simple helper to initialize data to None except in root Processes
        """
        return data if self.is_mpi_root() else None

    def bcast(self, data):
        """
        Broadcast data from the root MPI process to others.

        Arguments
        ---------
        data :
            Object to broadcast, not None on the root MPI process.
            Can be any python object. numpy arrays will be inefficient.

        Returns
        -------
        data :
            Broadcast item, on each process. Will match input from root.
        """
        # broadcast single items as themselves, not 1-element sequence
        if not self.mpi:
            return data
        return self.comm.bcast(data, root=self.mpi_root)

    def scatter(self, data):
        """
        Scatter parts of data from te root MPI process to others

        Arguments
        ---------
        data :
            Sequence of objects to scatter, must have one per MPI process.
            Can be any python object. numpy arrays will be inefficient.

        Returns
        -------
        data :
            Scattered items, on each process. Will match one element of the
            input from root.
        """
        if not self.mpi:
            # serially, must have a single element sequence. Return element
            if len(data) != 1:
                raise ValueError("Serialized scatter must have single element")
            return data[0]
        return self.comm.scatter(data, root=self.mpi_root)

    def gather(self, data):
        """
        Gather data from MPI process to root

        Arguments
        ---------
        data :
            Object to gather to the root process
            Can be any python object. numpy arrays will be inefficient.

        Returns
        -------
        data :
            On root, will be sequence of data gathered from each process.
        """
        if not self.mpi:
            # want to be able to iterate over returned results
            return [data]
        return self.comm.gather(data, root=self.mpi_root)

    # TODO add more efficient communication for np.arrays with mem buffer

    def mpi_local_size(self, size):
        """
        Return local size assuming the input size is divided
        evenly among processes.

        Arguments
        ---------
        size : int
            Size of sequence to split.

        Returns
        -------
        size_loc : int
            Size of local subset.
        """
        size_loc = size // self.mpi_size
        if self.mpi_rank < size % self.mpi_size:
            size_loc += 1
        return size_loc

    def mpi_local_index(self, size):
        """
        Return first local index into a sequence of the given size when
        it is divided evenly among processes.

        Arguments
        ---------
        size : int
            Size of sequence to split.

        Returns
        -------
        idx_loc : int
            Starting index of local subset
        """
        sz_loc = self.mpi_local_size(size)
        idx_loc = sz_loc * self.mpi_rank
        if self.mpi_rank >= size % self.mpi_size:
            idx_loc += size % self.mpi_size
        return idx_loc

    def mpi_local_range(self, size):
        """
        Return start and and indexes into a sequence of the given size when
        it is divided evenly among processes.

        Arguments
        ---------
        size : int
            Value to split.  Must be non-None on root process.

        Returns
        -------
        start_loc, end_loc : int
            Start and end indexes of local subset. (End is one above the last
            used index, as expected in range function.)
        """
        lstart = self.mpi_local_index(size)
        lsize = self.mpi_local_size(size)
        return lstart, lstart + lsize

if __name__ == "__main__":
    # Simple tests.
    # Note: despite barriers, log messages may appear out of order, due
    # to how stdout is buffered between processes.
    M = MPIHelper()
    M.log("Running some MPI tests", root=True)

    M.log("Hello from {} / {} {}".format(M.mpi_rank, M.mpi_size,
            "(root)" if M.is_mpi_root() else ""))
    M.barrier()

    # broadcast test
    data = M.none_except_root((14,15))
    M.log("Broadcast test sending {}".format(data), root=True)
    M.log("bcast  received {}".format(M.bcast(data)))
    M.barrier()

    # scatter test
    data = M.none_except_root(list(range(M.mpi_size)))
    M.log("Scatter test sending {}".format(data), root=True)
    M.log("scatter received {}".format(M.scatter(data)))
    M.barrier()

    # multi-element scatter test
    data = M.none_except_root(
            [[i+10*j for i in range(3)] for j in range(M.mpi_size)])
    M.log("Long Scatter test sending {}".format(data), root=True)
    M.log("Long scatter received {}".format(M.scatter(data)))
    M.barrier()

    # gather test
    M.log("Gather test", root=True)
    data = [(M.mpi_rank + 1 + i) * 11 for i in range(2)]
    M.log("Gather test sending {}".format(data))
    M.log("gather received {}".format(M.gather(data)), root=True)
    M.barrier()

    # range test
    M.log("Range test", root=True)
    size = 113
    start, end = M.mpi_local_range(size)
    M.log("Doing {} to {} of {}".format(start, end-1, size))
