package Util.Collections;

import java.util.Map;

/**
 * A <code>MultiMapSet</code> is a <code>java.util.Set</code> of
 * <code>Map.Entry</code>s which can also be accessed as a
 * <code>MultiMap</code>.  Use the <code>entrySet()</code> method
 * of the <code>MultiMap</code> to get back the <code>MultiMapSet</code>.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 */
public interface MultiMapSet/*<K,V>*/ extends MapSet/*<K,V>*/ {
    public Map/*<K,V>*/ asMap();
    public MultiMap/*<K,V>*/ asMultiMap();
}
