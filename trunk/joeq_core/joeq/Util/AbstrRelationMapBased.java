// AbstrRelationMapBased.java, created Fri Jun 30 11:17:10 2000 by salcianu
// Copyright (C) 2000 Alexandru SALCIANU <salcianu@MIT.EDU>
// Licensed under the terms of the GNU GPL; see COPYING for details.

package Util;

import java.io.Serializable;
import java.util.Collections;
import java.util.Map;
import java.util.Set;


/**
 * <code>AbstrRelationMapBased</code>
 * 
 * @author  Alexandru SALCIANU <salcianu@MIT.EDU>
 * @version $Id$
 */
public abstract class AbstrRelationMapBased extends AbstrRelation
    implements Serializable {
    
    // A map from keys to sets of values.
    protected Map map = null;
    

    public void removeKey(Object key) {
        hashCode = 0;
        map.remove(key);
    }

    
    public Set getValues(Object key) {
        Set retval = getValues2(key);
        if(retval == null)
            retval = Collections.EMPTY_SET;
        return retval;
    }

    protected Set getValues2(Object key) {      
        return (Set) map.get(key);
    }


    public Set keys() {
        return map.keySet();
    }
    
}
