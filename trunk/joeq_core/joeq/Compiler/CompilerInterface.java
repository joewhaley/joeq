/*
 * Compil3rInterface.java
 *
 * Created on December 16, 2000, 1:27 PM
 *
 */

package Compil3r;

import Clazz.jq_CompiledCode;
import Clazz.jq_Method;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface Compil3rInterface {
    jq_CompiledCode compile(jq_Method m);
}
