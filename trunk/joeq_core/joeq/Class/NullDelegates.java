package Clazz;

abstract class NullDelegates {
    static class Field implements jq_Field.Delegate {
	public final boolean isCodeAddressType(jq_Field f) { return false; }
	public final boolean isHeapAddressType(jq_Field f) { return false; }
	public final boolean isStackAddressType(jq_Field f) { return false; }
    }

    static class Method implements jq_Method.Delegate {
	public final jq_CompiledCode compile_stub (jq_Method m) {
	    return null;
	}
	public final jq_CompiledCode compile (jq_Method m) {
	    return null;
	}
    }
}
