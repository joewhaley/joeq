/*
 * UTFDataFormatError.java
 *
 * Created on December 19, 2000, 11:48 AM
 *
 * @author  jwhaley
 * @version 
 */

package UTF;

public class UTFDataFormatError extends RuntimeException {

    /**
     * Creates new <code>UTFDataFormatError</code> without detail message.
     */
    public UTFDataFormatError() {
    }

    /**
     * Constructs an <code>UTFDataFormatError</code> with the specified detail message.
     * @param msg the detail message.
     */
    public UTFDataFormatError(String msg) {
        super(msg);
    }
}
