/*
 * ObjectLayout.java
 *
 * Created on January 1, 2000, 10:15 PM
 *
 */

package Allocator;

/** This interface contains constants that define the joeq object layout.
 *  You can play with these constants to experiment with different object layouts.
 *
 * @author  John Whaley
 * @version $Id$
 */
public interface ObjectLayout {

    /**** OFFSETS ****/
    
    /** Offset of array length word, in bytes. */
    int ARRAY_LENGTH_OFFSET = -12;
    /** Offset of status word, in bytes. */
    int STATUS_WORD_OFFSET = -8;
    /** Offset of vtable, in bytes. */
    int VTABLE_OFFSET = -4;
    /** Offset of array element 0, in bytes. */
    int ARRAY_ELEMENT_OFFSET = 0;

    
    /**** HEADER SIZES ****/
    
    /** Size of (non-array) object header, in bytes. */
    int OBJ_HEADER_SIZE = 8;
    /** Size of array header, in bytes. */
    int ARRAY_HEADER_SIZE = 12;

    
    /**** STATUS BITS ****/
    
    /** Object has been hashed.  If it moves, we need to store the old address. */
    int HASHED       = 0x00000001;
    /** Object has been hashed and later moved.  The hash code is stored just past the object. */
    int HASHED_MOVED = 0x00000002;
    /** Bit in object header for use by GC. */
    int GC_BIT       = 0x00000004;
    /** Mask for status flags. */
    int STATUS_FLAGS_MASK = 0x00000007;

    
    /**** LOCKING ****/
    
    /** Bit location of thread id in the status word. */
    int THREAD_ID_SHIFT   = 9;
    /** Mask of the thread id in the status word. */
    int THREAD_ID_MASK    = 0x7FFFFE00;
    /** Mask of the lock count in the status word. */
    int LOCK_COUNT_MASK   = 0x000001F0;
    /** Value to add to status word to increment lock count by one. */
    int LOCK_COUNT_INC    = 0x00000010;
    /** Bit location of lock count in the status word. */
    int LOCK_COUNT_SHIFT  = 4;
    /** Lock has been expanded.
     *  Masking out this value and the status flags mask gives the address of the expanded lock structure. */
    int LOCK_EXPANDED     = 0x80000000;
    
}
