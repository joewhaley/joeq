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
}
