package ClassLib.Common;

import Clazz.jq_Member;
import Clazz.jq_Class;
import Clazz.jq_CompiledCode;
import Memory.StackAddress;
import Run_Time.StackCodeWalker;
import Run_Time.TypeCheck;

public abstract class ClassUtils {
    public static final void checkCallerAccess(jq_Member m, int depth) throws IllegalAccessException {
	jq_Class field_class = m.getDeclaringClass();
	if (m.isPublic() && field_class.isPublic()) {
	    // completely public!
	    return;
	}
	StackCodeWalker sw = new StackCodeWalker(null, StackAddress.getBasePointer());
	while (--depth >= 0) sw.gotoNext();
	jq_CompiledCode cc = sw.getCode();
	if (cc != null) {
	    jq_Class caller_class = cc.getMethod().getDeclaringClass();
	    if (caller_class == field_class) {
		// same class! access allowed!
		return;
	    }
	    if (field_class.isPublic() || caller_class.isInSamePackage(field_class)) {
		if (m.isPublic()) {
		    // class is accessible and field is public!
		    return;
		}
		if (m.isProtected()) {
		    if (TypeCheck.isAssignable(caller_class, field_class)) {
			// field is protected and field_class is supertype of caller_class!
			return;
		    }
		}
		if (!m.isPrivate()) {
		    if (caller_class.isInSamePackage(field_class)) {
			// field is package-private and field_class and caller_class are in the same package!
			return;
		    }
		}
	    }
	}
	throw new IllegalAccessException();
    }
}
