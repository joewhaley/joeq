/*
 * Interface.java
 *
 * Created on December 12, 2001, 1:27 AM
 *
 */

package ClassLib.sun13_win32;

import java.util.Iterator;

import Bootstrap.ObjectTraverser;
import ClassLib.ClassLibInterface;
import Scheduler.jq_NativeThread;

/*
 * @author  John Whaley
 * @version $Id$
 */
public final class Interface extends ClassLib.Common.Interface {

    /** Creates new Interface */
    public Interface() {}

    public Iterator getImplementationClassDescs(UTF.Utf8 desc) {
        if (ClassLibInterface.USE_JOEQ_CLASSLIB && desc.toString().startsWith("Ljava/")) {
            UTF.Utf8 u = UTF.Utf8.get("LClassLib/sun13_win32/"+desc.toString().substring(1));
            return new Util.AppendIterator(super.getImplementationClassDescs(desc),
                                            java.util.Collections.singleton(u).iterator());
        }
        return super.getImplementationClassDescs(desc);
    }
    
    public ObjectTraverser getObjectTraverser() {
    	return sun13_win32ObjectTraverser.INSTANCE;
    }
    
    public static class sun13_win32ObjectTraverser extends CommonObjectTraverser {
    	public static sun13_win32ObjectTraverser INSTANCE = new sun13_win32ObjectTraverser();
    	protected sun13_win32ObjectTraverser() {}
        public void initialize() {
            super.initialize();
            jq_NativeThread.USE_INTERRUPTER_THREAD = true;
        }
    }
    
}
