/*
 * DefaultCodeAllocator.java
 *
 * Created on April 1, 2001, 5:59 PM
 */

package Allocator;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_StaticField;
import Run_Time.Unsafe;
import Allocator.CodeAllocator.x86CodeBuffer;

/*
 *
 * @author  John Whaley
 * @version $Id$
 */
public abstract class DefaultCodeAllocator {

    public static CodeAllocator default_allocator;

    public static final CodeAllocator def() {
        if (default_allocator != null) return default_allocator;
        return Unsafe.getThreadBlock().getNativeThread().getCodeAllocator();
    }
    
    public static final void init() {
        def().init();
    }
    public static final x86CodeBuffer getCodeBuffer(int estimated_size) {
        x86CodeBuffer o = def().getCodeBuffer(estimated_size);
        return o;
    }
    public static final void patchAbsolute(int/*CodeAddress*/ code, int/*HeapAddress*/ heap) {
        def().patchAbsolute(code, heap);
    }
    public static final void patchRelativeOffset(int/*CodeAddress*/ code, int/*CodeAddress*/ target) {
        def().patchRelativeOffset(code, target);
    }
    
    public static final jq_StaticField _default_allocator;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LAllocator/DefaultCodeAllocator;");
        _default_allocator = k.getOrCreateStaticField("default_allocator", "LAllocator/CodeAllocator;");
    }
}
