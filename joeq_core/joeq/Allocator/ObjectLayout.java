/*
 * ObjectLayout.java
 *
 * Created on January 1, 2000, 10:15 PM
 *
 * @author  jwhaley
 * @version 
 */

package Allocator;

public interface ObjectLayout {

    public static final int ARRAY_LENGTH_OFFSET = -12;
    public static final int STATUS_WORD_OFFSET = -8;
    public static final int VTABLE_OFFSET = -4;
    public static final int ARRAY_ELEMENT_OFFSET = 0;

    public static final int OBJ_HEADER_SIZE = 8;
    public static final int ARRAY_HEADER_SIZE = 12;

    public static final int HASHED       = 0x00000001;
    public static final int HASHED_MOVED = 0x00000002;
    
    public static final int STATUS_FLAGS_MASK = 0x00000007;

    public static final int THREAD_ID_SHIFT   = 9;
    public static final int THREAD_ID_MASK    = 0x7FFFFE00;
    public static final int LOCK_COUNT_MASK   = 0x000001F0;
    public static final int LOCK_COUNT_INC    = 0x00000010;
    public static final int LOCK_EXPANDED     = 0x80000000;
}
