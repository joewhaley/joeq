/*
 * jq_DontAlign.java
 *
 * Created on April 3, 2001, 8:29 PM
 *
 */

package Clazz;

import Bootstrap.PrimordialClassLoader;

/**
 * This interface is used as a marker to signify that the fields in the
 * class should not be aligned.  This is necessary if the layout must
 * match another data structure exactly, for example: in
 * Scheduler.jq_RegisterState.
 * 
 * @see Scheduler.jq_RegisterState
 * @author  John Whaley
 * @version $Id$
 */
public interface jq_DontAlign {

    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LClazz/jq_DontAlign;");
}
