package Memory.Heap;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import Allocator.ObjectLayoutMethods;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Memory.HeapAddress;
import Run_Time.Debug;
import Run_Time.SystemInterface;
import Util.Assert;

/**
 * This is a reference to an abstract memory "heap".
 * 
 * This class also includes static methods that operate on all heaps.
 * 
 * @author John Whaley
 */
public abstract class Heap {

    /**
     * Reference to the heap present at boot up time.
     * This is the memory that is mapped in from the data section
     * of the executable.
     */
    //public static final BootHeap bootHeap = BootHeap.INSTANCE;
    
    /**
     * Reference to the initial malloc heap.
     */
    //public static final MallocHeap mallocHeap = new MallocHeap();
    
    /**
     * List of all heaps in the system.
     */
    public static final List/*<Heap>*/ allHeaps = new ArrayList();

    /**
     * Name of this heap.
     */
    protected final String name;

    /**
     * Amount of chattering during operation.
     */
    protected int verbose = 0;

    /**
     * Start of range of memory belonging to this heap, inclusive.
     */
    protected HeapAddress start;
    
    /**
     * End of range of memory belonging to this heap, exclusive.
     */
    protected HeapAddress end;
    
    /*
     * Static methods that manipulate all heaps
     */

    /**
     * Initialize the default heaps.
     * Must be called exactly once at startup.
     */
    public static void boot() {
        BootHeap.init();
        //mallocHeap.init();
        Assert._assert(BootHeap.INSTANCE.refInHeap(HeapAddress.addressOf(BootHeap.INSTANCE)));
    }

    /**
     * Return true if the given reference is to an 
     * object that is located within some heap.
     */
    public static boolean refInAnyHeap(HeapAddress ref) {
        for (Iterator i=allHeaps.iterator(); i.hasNext(); ) {
            Heap h = (Heap) i.next();
            if (h.refInHeap(ref)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Return true if the given address is located within some heap.
     */
    public static boolean addrInAnyHeap(HeapAddress addr) {
        for (Iterator i=allHeaps.iterator(); i.hasNext(); ) {
            Heap h = (Heap) i.next();
            if (h.addrInHeap(addr)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Ask all known heaps to show themselves
     */
    public static void showAllHeaps() {
        Debug.writeln(allHeaps.size(), " heaps");
        int j=0;
        for (Iterator i=allHeaps.iterator(); i.hasNext(); ++j) {
            Debug.write("Heap ", j, ": ");
            Heap h = (Heap) i.next();
            h.show();
        }
    }

    /**
     * Clobber the specified address range; useful for debugging
     */
    public static void clobber(HeapAddress start, HeapAddress end) {
        Debug.write("Zapping region ", start);
        Debug.write(" .. ", end);
        Debug.writeln(" with 0xff****ff: ");
        int size = end.difference(start);
        for (int i = 0; i < size; i += 4) {
            int pattern = 0xff0000ff;
            pattern |= i & 0x00ffff00;
            start.offset(i).poke4(pattern);
        }
    }

    /*
     * Heap instance methods
     */

    /**
     * Constructor.  Also adds heap to the allHeaps list.
     */
    protected Heap(String n) {
        name = n;
        start = end = HeapAddress.getNull();
        allHeaps.add(this);
    }

    /**
     * Set the region that the heap is managing.
     */
    public void setRegion(HeapAddress s, HeapAddress e) {
        start = s;
        end = e;
    }

    /**
     * Size of the heap in bytes.
     */
    public int getSize() {
        return end.difference(start);
    }

    /**
     * Zero the entire heap.
     */
    public void zero() {
        // verify page alignment.
        Assert._assert(start.difference(start.align(HeapAddress.pageAlign())) == 0);
        Assert._assert(end.difference(end.align(HeapAddress.pageAlign())) == 0);
        int size = getSize();
        SystemInterface.mem_set(start, size, (byte)0);
    }

        /*
    public void grow(int sz) {
        if (sz < size)
            jq.UNREACHABLE("Heap.grow given smaller size than current size\n");
        sz = Address.align(sz, HeapAddress.pageAlign());
        int flag1 = SystemInterface.PROT_READ |
                    SystemInterface.PROT_WRITE |
                    SystemInterface.PROT_EXEC;
        int flag2 = SystemInterface.MAP_PRIVATE |
                    SystemInterface.MAP_ANONYMOUS |
                    SystemInterface.MAP_FIXED;
        HeapAddress result = SystemInterface.mmap(end, sz-size, flag1, flag2);
        if (result.isNull()) {
            Debug.write("Heap.grow failed to mmap additional ");
            Debug.write((sz - size) / 1024);
            Debug.writeln(" Kbytes at ", end);
            SystemInterface.die(-1);
            jq.UNREACHABLE();
        }
        if (verbose >= 1) {
            Debug.write("Heap.grow successfully mmap additional ");
            Debug.write((sz - size) / 1024);
            Debug.writeln(" Kbytes at ", end);
        }
        // start not modified
        end = (HeapAddress) start.offset(sz);
    }
        */

    public boolean refInHeap(HeapAddress ref) {
        return addrInHeap(ref);
    }

    public boolean addrInHeap(HeapAddress addr) {
        return addr.difference(start) >= 0 && addr.difference(end) <= 0;
    }

    public void showRange() {
        Debug.write(start);
        Debug.write(" .. ", end);
    }

    public void show() {
        show(true);
    }

    public void show(boolean newline) {
        int tab = 26 - name.length();
        for (int i = 0; i < tab; i++)
            Debug.write(" ");
        Debug.write(name);
        int size = getSize();
        Debug.write(": ", size / 1024);
        Debug.write(" Kb  at  ");
        showRange();
        if (newline)
            Debug.writeln();
    }

    public void touchPages() {
        int ps = 1 << HeapAddress.pageAlign();
        int size = getSize();
        for (int i = size - ps; i >= 0; i -= ps)
            start.offset(i).poke4(0);
    }

    public void clobber() {
        clobber(start, end);
    }

    /**
     * Scan this heap for references to the target heap and report them.
     * This is approximate since the scan is done without type information.
     */
    public final int paranoidScan(Heap target, boolean show) {
        int count = 0;
        Debug.write("Checking heap ");
        showRange();
        Debug.write(" for references to ");
        target.showRange();
        Debug.writeln();
        for (HeapAddress loc = start; loc.difference(end) < 0; loc = (HeapAddress) loc.offset(HeapAddress.size())) {
            HeapAddress value = (HeapAddress) loc.peek();
            if (((value.to32BitValue() & 3) == 0) && target.refInHeap(value)) {
                count++;
                if (show) {
                    Debug.write("Warning # ", count);
                    Debug.write("  loc ", loc);
                    Debug.writeln(" holds poss ref ", value);
                }
            }
        }
        Debug.write("\nThere were ", count);
        Debug.write(" suspicious references to ");
        target.showRange();
        Debug.writeln();
        return count;
    }

    /**
     * Allocate a scalar object. Fills in the header for the object
     * and sets all data fields to zero. Assumes that type is already initialized.
     * 
     * @param type  jq_Class of type to be instantiated
     *
     * @return the reference for the allocated object
     */
    public final Object allocateObject(jq_Class type) {
        Assert._assert(type.isClsInitialized());
        int size = type.getInstanceSize();
        Object tib = type.getVTable();
        return allocateObject(size, tib);
    }

    /**
     * Allocate an array object. Fills in the header for the object,
     * sets the array length to the specified length, and sets
     * all data fields to zero.  Assumes that type is already initialized.
     *
     * @param type  jq_Array of type to be instantiated
     * @param numElements  number of array elements
     *
     * @return the reference for the allocated array object 
     */
    public final Object allocateArray(jq_Array type, int numElements) {
        Assert._assert(type.isClsInitialized());
        int size = type.getInstanceSize(numElements);
        Object tib = type.getVTable();
        return allocateArray(numElements, size, tib);
    }

    /**
     * Allocate a scalar object. Fills in the header for the object,
     * and set all data fields to zero.
     *
     * @param size         size of object (including header), in bytes
     * @param tib          type information block for object
     *
     * @return the reference for the allocated object
     */
    public final Object allocateObject(int size, Object vtable)
        throws OutOfMemoryError {
        HeapAddress region = allocateZeroedMemory(size);
        Object newObj = ObjectLayoutMethods.initializeObject(region, vtable, size);
        postAllocationProcessing(newObj);
        return newObj;
    }

    /**
     * Allocate an array object. Fills in the header for the object,
     * sets the array length to the specified length, and sets
     * all data fields to zero.
     *
     * @param numElements  number of array elements
     * @param size         size of array object (including header), in bytes
     * @param tib          type information block for array object
     *
     * @return the reference for the allocated array object 
     */
    public final Object allocateArray(int numElements, int size, Object vtable)
        throws OutOfMemoryError {
        // note: array size might not be a word multiple,
        //       must preserve alignment of future allocations.
        size = HeapAddress.align(size, HeapAddress.logSize());
        HeapAddress region = allocateZeroedMemory(size);
        Object newObj = ObjectLayoutMethods.initializeArray(region, vtable, numElements, size);
        postAllocationProcessing(newObj);
        return newObj;
    }

    /**
     * Allocate size bytes of zeroed memory.
     * Size is a multiple of wordsize, and the returned memory must be word aligned
     * 
     * @param size Number of bytes to allocate
     * @return Address of allocated storage
     */
    protected abstract HeapAddress allocateZeroedMemory(int size);

    /**
     * Hook to allow heap to perform post-allocation processing of the object.
     * For example, setting the GC state bits in the object header.
     * 
     * @param newObj the object just allocated in the heap.
     */
    protected abstract void postAllocationProcessing(Object newObj);
}
