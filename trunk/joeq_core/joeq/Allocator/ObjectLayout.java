/*
 * ObjectLayout.java
 *
 * Created on January 1, 2000, 10:15 PM
 *
 */

package Allocator;


import Memory.HeapAddress;
import Run_Time.Unsafe;

/** This interface contains constants that define the joeq object layout.
 *  You can play with these constants to experiment with different object layouts.
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class ObjectLayout {

    /**** OFFSETS ****/
    
    /** Offset of array length word, in bytes. */
    public static final int ARRAY_LENGTH_OFFSET = -12;
    /** Offset of status word, in bytes. */
    public static final int STATUS_WORD_OFFSET = -8;
    /** Offset of vtable, in bytes. */
    public static final int VTABLE_OFFSET = -4;
    /** Offset of array element 0, in bytes. */
    public static final int ARRAY_ELEMENT_OFFSET = 0;

    
    /**** HEADER SIZES ****/
    
    /** Size of (non-array) object header, in bytes. */
    public static final int OBJ_HEADER_SIZE = 8;
    /** Size of array header, in bytes. */
    public static final int ARRAY_HEADER_SIZE = 12;

    
    /**** STATUS BITS ****/
    
    /** Object has been hashed.  If it moves, we need to store the old address. */
    public static final int HASHED       = 0x00000001;
    /** Object has been hashed and later moved.  The hash code is stored just past the object. */
    public static final int HASHED_MOVED = 0x00000002;
    /** Bit in object header for use by GC. */
    public static final int GC_BIT       = 0x00000004;
    /** Mask for status flags. */
    public static final int STATUS_FLAGS_MASK = 0x00000007;

    
    /**** LOCKING ****/
    
    /** Bit location of thread id in the status word. */
    public static final int THREAD_ID_SHIFT   = 9;
    /** Mask of the thread id in the status word. */
    public static final int THREAD_ID_MASK    = 0x7FFFFE00;
    /** Mask of the lock count in the status word. */
    public static final int LOCK_COUNT_MASK   = 0x000001F0;
    /** Value to add to status word to increment lock count by one. */
    public static final int LOCK_COUNT_INC    = 0x00000010;
    /** Bit location of lock count in the status word. */
    public static final int LOCK_COUNT_SHIFT  = 4;
    /** Lock has been expanded.
     *  Masking out this value and the status flags mask gives the address of the expanded lock structure. */
    public static final int LOCK_EXPANDED     = 0x80000000;
    
    
    /**** UTILITY FUNCTIONS ****/

    public static Object initializeObject(HeapAddress addr, Object vtable, int size) {
        addr = (HeapAddress) addr.offset(OBJ_HEADER_SIZE);
        addr.offset(VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    public static Object initializeArray(HeapAddress addr, Object vtable, int length, int size) {
        addr = (HeapAddress) addr.offset(ARRAY_HEADER_SIZE);
        addr.offset(ARRAY_LENGTH_OFFSET).poke4(length);
        addr.offset(VTABLE_OFFSET).poke(HeapAddress.addressOf(vtable));
        return addr.asObject();
    }
    
    public static int getArrayLength(Object obj) {
        HeapAddress addr = HeapAddress.addressOf(obj);
        return addr.offset(ARRAY_LENGTH_OFFSET).peek4();
    }
    
    public static void setArrayLength(Object obj, int newLength) {
        HeapAddress addr = HeapAddress.addressOf(obj);
        addr.offset(ARRAY_LENGTH_OFFSET).poke4(newLength);
    }
    
    public static Object getVTable(Object obj) {
        HeapAddress addr = HeapAddress.addressOf(obj);
        return ((HeapAddress) addr.offset(VTABLE_OFFSET).peek()).asObject();
    }
    
    public static boolean testAndMark(Object obj, int markValue) {
        HeapAddress addr = (HeapAddress) HeapAddress.addressOf(obj).offset(STATUS_WORD_OFFSET);
        for (;;) {
            int oldValue = addr.peek4();
            int newValue = (oldValue & ~GC_BIT) | markValue;
            if (oldValue == newValue)
                return false;
            addr.atomicCas4(oldValue, newValue);
            if (Unsafe.isEQ())
                break;
        }
        return true;
    }
    
    public static boolean testMarkBit(Object obj, int markValue) {
        HeapAddress addr = (HeapAddress) HeapAddress.addressOf(obj).offset(STATUS_WORD_OFFSET);
        int value = addr.peek4();
        return (value & GC_BIT) == markValue;
    }
    
    public static void writeMarkBit(Object obj, int markValue) {
        HeapAddress addr = (HeapAddress) HeapAddress.addressOf(obj).offset(STATUS_WORD_OFFSET);
        int oldValue = addr.peek4();
        int newValue = (oldValue & ~GC_BIT) | markValue;
        addr.poke4(newValue);
    }
}
