/*
 * Strings.java
 *
 * Created on October 25, 2001, 12:01 PM
 *
 * @author  jwhaley
 * @version 
 */

package Util;

public abstract class Strings {

    /**
     * Replace all occurences of <em>old</em> in <em>str</em> with <em>new</em>.
     *
     * @param str String to permute
     * @param old String to be replaced
     * @param new Replacement string
     * @return new String object
     */
    public static final String replace(String str, String old, String new_) {
        int          index, old_index;
        StringBuffer buf = new StringBuffer();
        
        try {
            if((index = str.indexOf(old)) != -1) { // `old' found in str
                old_index = 0;                       // String start offset
                
                // While we have something to replace
                while((index = str.indexOf(old, old_index)) != -1) {
                    buf.append(str.substring(old_index, index)); // append prefix
                    buf.append(new_);                            // append replacement
                    
                    old_index = index + old.length(); // Skip `old'.length chars
                }
                
                buf.append(str.substring(old_index)); // append rest of string
                str = buf.toString();
            }
        } catch(StringIndexOutOfBoundsException e) { // Should not occur
            System.err.println(e);
        }
        
        return str;
    }

    /**
     * Return a string for an integer justified left or right and filled up with
     * `fill' characters if necessary.
     *
     * @param i integer to format
     * @param length length of desired string
     * @param left_justify format left or right
     * @param fill fill character
     * @return formatted int
     */
    public static final String format(int i, int length, boolean left_justify, char fill) {
        return fillup(Integer.toString(i), length, left_justify, fill);
    }

    /**
     * Fillup char with up to length characters with char `fill' and justify it left or right.
     *
     * @param str string to format
     * @param length length of desired string
     * @param left_justify format left or right
     * @param fill fill character
     * @return formatted string
     */
    public static final String fillup(String str, int length, boolean left_justify, char fill) {
        int    len = length - str.length();
        char[] buf = new char[(len < 0)? 0 : len];

        for(int j=0; j < buf.length; j++)
            buf[j] = fill;

        if(left_justify)
            return str + new String(buf);    
        else
            return new String(buf) + str;
    }

}
