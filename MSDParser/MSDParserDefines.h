/////////////////////////////////////////////////////////////////////////
// Defines (MSDParser Module)
// ------------------
// Definitions used by the MSDParser Module.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_MSDParser_Defines
#define MOD_MSDParser_Defines

namespace MSDParser
{
	enum class ObjectGroup
	{
		SpinSystem,
		Settings,
		Run,
		None, // FileReader is not currently within an object group
	};

	enum class ObjectType
	{
		Spin = 0,
		Interaction,
		Transition,
		Operator,
		State,
		Pulse,
		PulseSequence,
		SubSystem,
		Output,
		Task,
		Action,
		Settings,
		Properties,
		Undefined, // Indicates that no valid object was read (e.g. nothing read yet, or a read object is of an unknown type)
		EOFObject, // Indicates that there are no more objects in the input file
	};

	struct MSDFileObject
	{
	private:
		ObjectType type;
		std::string name;
		std::string contents;

	public:
		MSDFileObject(ObjectType _type = ObjectType::EOFObject, std::string _name = "", std::string _contents = "") : type(_type), name(_name), contents(_contents){};
		MSDFileObject(const MSDFileObject &_obj) : type(_obj.type), name(_obj.name), contents(_obj.contents){};

		ObjectType Type() const { return type; };
		std::string Name() const { return name; };
		std::string Contents() const { return contents; };
	};
}

#endif
