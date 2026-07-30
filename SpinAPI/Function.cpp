/////////////////////////////////////////////////////////////////////////
// Function implementation (SpinAPI Module)
// ------------------
// The Function class represents a mathmatical function.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

#include <cctype>
#include <cmath>
#include <memory>
#include <armadillo>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>

#include "Function.h"
namespace SpinAPI
{

	static int s_ComplexNum = 0;
	// Function class stuff
	arma::cx_double Function::operator()(void *value)
	{
		arma::cx_double ReturnValue;

		if (m_duplicates.size() != 0) // if there's duplicates but only one variable the overload gets called
		{
			std::vector<void *> values;
			for (unsigned int i = 0; i < m_duplicates.size(); i++)
			{
				values.push_back(value);
			}
			return this->operator()(values);
		}

		if (m_funcType == ReturnType::d)
		{
			std::complex<double> val = EvaluateFuncValue({value}); //
			double *temp = (double *)m_func((void *)(std::complex<double> *)&val);
			double _val = *temp;
			delete temp;
			ReturnValue = arma::cx_double(_val, 0);
		}
		else if (m_funcType == ReturnType::cd)
		{
			std::complex<double> val = EvaluateFuncValue({value});
			arma::cx_double *temp = (arma::cx_double *)m_func((void *)(std::complex<double> *)&val);
			arma::cx_double _val = *temp;
			delete temp;
			ReturnValue = _val;
		}

		return ReturnValue;
	}

	arma::cx_double Function::operator()(std::vector<void *> value)
	{
		arma::cx_double ReturnValue;
		if (m_funcType == ReturnType::d)
		{
			std::complex<double> val = EvaluateFuncValue(value);
			double *temp = (double *)m_func((void *)(std::complex<double> *)&val);
			double _val = *temp;
			delete temp;
			ReturnValue = arma::cx_double(_val, 0);
		}
		else if (m_funcType == ReturnType::cd)
		{
			std::complex<double> val = EvaluateFuncValue(value);
			arma::cx_double *temp = (arma::cx_double *)m_func((void *)(std::complex<double> *)&val);
			arma::cx_double _val = *temp;
			delete temp;
			ReturnValue = _val;
		}

		return ReturnValue;
	}

	std::complex<double> Function::EvaluateFuncValue(std::vector<void *> values)
	{
		std::vector<double> VarValues;
		std::vector<unsigned int> VarSize;
		std::vector<std::pair<std::string, std::complex<double>>> TopLevelVariables;
		std::string AllocatedVariables = "";
		std::unordered_map<std::string, std::complex<double>> VarListTemp;
		std::unordered_map<std::string, std::complex<double>> VarList;
		for (unsigned int i = 0; i < m_variables.size(); i++)
		{
			std::vector<std::string> temp = m_variables[i].InternalVariables;
			TopLevelVariables.push_back({m_variables[i].name, std::complex<double>(0)});
			for (unsigned int e = 0; e < temp.size(); e++)
			{
				if (temp[e] == "")
				{
					VarValues.push_back(1.0); // if the variable is "" (i.e no variable exists) it just puts a 1 in its place otherwise it puts 0 in as a placeholder value
				}
				else
				{
					VarValues.push_back(0.0);
					VarListTemp.insert({temp[e], 0.0});
				}

				if (AllocatedVariables.find(temp[e]) != AllocatedVariables.npos)
					continue;

				AllocatedVariables = AllocatedVariables + temp[e];
			}
			VarSize.push_back(temp.size());
		}

		for (auto v = VarListTemp.begin(); v != VarListTemp.end(); v++)
		{
			VarList.insert({v->first, v->second});
		}

		int index = 0;
		for (auto v = values.begin(); v != values.end(); v++)
		{
			if (VarValues[index] != 0.0)
			{
				v--;
				index++;
				continue;
			}

			VarValues[index] = *(double *)(*v); // gets the variable value;
			auto Varit = VarList.begin();
			std::advance(Varit, index);

			Varit->second = VarValues[index];
		}

		std::vector<std::string> AllVariables;
		for (auto v = this->m_variables.begin(); v != this->m_variables.end(); v++)
		{
			for (auto a = v->InternalVariables.begin(); a != v->InternalVariables.end(); a++)
			{
				AllVariables.push_back((*a));
			}
		}

		index = 0;
		std::vector<double> VarValuesTemp;
		for (auto v = AllVariables.begin(); v != AllVariables.end(); v++)
		{
			VarValuesTemp.push_back(VarList[(*v)].real());
		}
		VarValues = VarValuesTemp;

		auto Evaluate = [](std::unordered_map<std::string, std::complex<double>> ValueMap, std::string PostFixEQ)
		{
			std::stack<std::complex<double>> ValueStack;
			std::complex<double> Val1(0);
			std::complex<double> Val2(0);
			const std::complex<double> invalid(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
			std::vector<char> chr = {'+', '-', '*', '^', '/', '(', ')'};
			bool Float = false;
			for (unsigned int i = 0; i < PostFixEQ.size();)
			{
				char c = PostFixEQ[i];
				auto it = std::find(chr.begin(), chr.end(), c);
				if (it == chr.end())
				{
					std::string VarName = "";
					if (std::isdigit(c) != 0)
					{
						Float = true;
					}
					if (Float)
					{
						while (i < PostFixEQ.size() && (std::isdigit(c) || c == '.' || c == ',') && std::find(chr.begin(), chr.end(), c) == chr.end())
						{
							VarName += c;
							i++;
							if (i >= PostFixEQ.size())
								break;
							c = PostFixEQ[i];
						}
						i--;
						ValueStack.push(std::stod(VarName));
						Float = false;
					}
					else
					{
						bool exist = true;
						bool found = false;
						while (exist && !found)
						{
							for (auto key = ValueMap.begin(); key != ValueMap.end(); key++)
							{
								if (key->first.find(VarName) != std::string::npos)
								{
									if (VarName == key->first)
									{
										found = true;
										break;
									}
									exist = true;
									break;
								}
							}
							if (!found)
							{
								VarName += c;
								i++;
								if (i >= PostFixEQ.size())
									break;
								c = PostFixEQ[i];
							}
						}
						ValueStack.push(ValueMap[VarName]);
						i--;
					}
				}
				else
				{
					if (ValueStack.empty())
						return invalid;
					Val2 = ValueStack.top();
					ValueStack.pop();
					if (ValueStack.size() > 0)
					{
						Val1 = ValueStack.top();
						ValueStack.pop();
					}
					else
					{
						Val1 = std::complex<double>(0);
					}
					std::complex<double> total;
					switch (c)
					{
					case '+':
						total = Val1 + Val2;
						break;
					case '-':
						total = Val1 - Val2;
						break;
					case '*':
						total = Val1 * Val2;
						break;
					case '/':
						total = Val1 / Val2;
						break;
					case '^':
						total = std::pow(Val1, Val2);
						break;
					default:
						total = 1.0;
						break;
					}
					ValueStack.push(total);
				}
				i++;
			}
			if (ValueStack.empty())
				return invalid;
			return ValueStack.top();
		};
		index = 0;
		int index2 = 0;

		auto findz = [&index, &TopLevelVariables](VariableDefinition &v2)
		{
			if (v2.name == TopLevelVariables[index].first)
			{
				return true;
			}
			return false;
		};

		for (auto v = m_variables.begin(); v != m_variables.end(); v++)
		{
			if (v->type == VarType::d)
			{
				TopLevelVariables[index].second = std::complex<double>(VarValues[index2], 0);
				index2++;
			}
			else if (v->type == VarType::f)
			{
				std::vector<void *> val;
				auto vdef = std::find_if(m_variables.begin(), m_variables.end(), findz);
				for (int i = 0; i < int(vdef->InternalVariables.size()); i++)
				{
					val.push_back((void *)&VarValues[index2]);
					index2++;
				}
				TopLevelVariables[index].second = vdef->InternalFunction->operator()(val); //->InternalFunction->EvaluateFuncValue(val);
			}
			else if (v->type == VarType::z)
			{
				auto vdef = std::find_if(m_variables.begin(), m_variables.end(), findz);

				std::string str1 = "";
				std::string str2 = "";

				bool comma = false;
				for (unsigned int i = 0; i < vdef->VariableString.size(); i++)
				{
					char c = vdef->VariableString[i];
					if (c == '#')
					{
						comma = true;
						continue;
					}
					if (!comma)
					{
						str1 = str1 + c;
					}
					else
					{
						str2 = str2 + c;
					}
				}
				auto v1 = Evaluate(VarList, str1);
				auto v2 = Evaluate(VarList, str2);
				TopLevelVariables[index].second = std::complex<double>(v1.real(), v2.real());
				index2 = index2 + vdef->InternalVariables.size();
			}
			index++;
		}

		std::unordered_map<std::string, std::complex<double>> TopLevelMap;
		for (auto a = TopLevelVariables.begin(); a != TopLevelVariables.end(); a++)
		{
			TopLevelMap.insert({a->first, a->second});
		}

		return Evaluate(TopLevelMap, m_PostFixEquation);
	}

	Function::Function(FuncPtr FunctionPtr, ReturnType type, std::string name, std::vector<VariableDefinition> vars)
	{
		m_func = FunctionPtr;
		m_funcType = type;
		m_FunctionName = name;
		m_variables = vars;
		m_FunctionString = "";
		m_PostFixEquation = "";

		std::vector<std::string> TempVar;
		for (auto v = m_variables.cbegin(); v != m_variables.cend(); v++)
		{
			for (auto e = v->InternalVariables.cbegin(); e != v->InternalVariables.cend(); e++)
			{
				if ((*e) == "")
				{
					continue;
				}

				if (TempVar.size() == 0)
				{
					TempVar.push_back((*e));
					continue;
				}

				if (std::find(TempVar.begin(), TempVar.end(), (*e)) == TempVar.end())
				{
					TempVar.push_back((*e));
				}
				else if (std::find(m_duplicates.begin(), m_duplicates.end(), (*e)) == m_duplicates.end())
				{
					m_duplicates.push_back((*e));
				}
			}
		}
	}

	Function::Function(FuncPtr FunctionPtr, ReturnType type, std::string name, VariableDefinition def)
	{
		m_func = FunctionPtr;
		m_funcType = type;
		m_FunctionName = name;
		m_variables = {def};
		m_FunctionString = "";
		m_PostFixEquation = "";
	}

	std::vector<std::string> Function::GetVariable()
	{
		std::vector<std::string> variables;
		std::unordered_set<std::string> TempSet;

		std::vector<std::string> AllVariables;
		for (auto v : m_variables)
		{
			for (auto v2 : v.InternalVariables)
			{
				AllVariables.push_back(v2);
			}
		}

		TempSet.insert(AllVariables.begin(), AllVariables.end());
		for (auto var : TempSet)
		{
			variables.push_back(var);
		}

		std::reverse(variables.begin(), variables.end());

		return variables;
	}

	std::string Function::GetName()
	{
		return m_FunctionName;
	}

	Function::VariableDefinition Function::FindVar(std::string var)
	{
		auto FinnVarDef = [&var](const VariableDefinition &def)
		{
			return (var == def.name);
		};

		auto it = std::find_if(m_variables.cbegin(), m_variables.cend(), FinnVarDef);
		if (it == m_variables.cend())
		{
			return {"", VarType::d, "", {""}, nullptr};
		}
		return (*it);
	}

	static std::vector<std::pair<std::string, bool>> s_MathmaticalFunctions = {{"cos", 0}, {"sin", 0}, {"scalar", 0}, {"scalarcx", 1}, {"exp", 0}, {"expcx", 1}};
	std::shared_ptr<Function> FunctionParser(std::string &func, std::string &var, int FuncStartNum, bool ResetComplexNum)
	{
		std::string FunctionString = var;
		if (FunctionString.empty())
			return nullptr;

		std::vector<char> SpecialCharacters = {'+', '-', '*', '/', '^', '(', ')'};
		std::vector<char> IgnoreCharacters = {'.', 'i', ',', 'j'}; // characters to ignore as part of text i.e 3+4ix gets read as 3 + (4i * x) not 3 + 4 * ix
		std::vector<std::string> MathematicalFunctionsList;
		for (auto a = s_MathmaticalFunctions.cbegin(); a != s_MathmaticalFunctions.cend(); a++)
		{
			MathematicalFunctionsList.push_back(a->first);
		}

		std::string buffer = "";
		// std::vector<char>::iterator it = SpecialCharacters.end();
		// std::vector<std::string>::iterator it2 = MathematicalFunctionsList.end();

		if (ResetComplexNum)
		{
			s_ComplexNum = 0;
		}

		auto prec = [](char c)
		{
			if (c == '^')
				return 3;
			else if (c == '/' || c == '*')
				return 2;
			else if (c == '+' || c == '-')
				return 1;
			else
				return -1;
		};

		auto CheckFloat = [](std::string buffer)
		{
			std::string buffer1 = "";
			std::string buffer2 = "";
			bool decimal = false;
			bool IsFloat = true;

			for (auto c = buffer.cbegin(); c != buffer.cend(); c++)
			{
				if ((*c) == '.' || (*c) == ',')
				{
					decimal = true;
				}
				else if (std::isdigit((*c)) != 0)
				{
					(!decimal) ? buffer1 += (*c) : buffer2 += (*c);
				}
				else if ((*c) == 'i' || (*c) == 'j')
				{
					continue;
				}
				else
				{
					IsFloat = false;
					break;
				}
			}

			if (!IsFloat)
			{
				return false;
			}

			return true;
		};

		// auto TextOnly = [](std::string buffer) {
		// 	if(buffer.length() == 1)
		// 	{
		// 		if(buffer == "i" || buffer == "j")
		// 		{
		// 			return false;
		// 		}
		// 	}
		// 	for(auto c = buffer.begin(); c != buffer.end(); c++)
		// 	{
		// 		if(std::isdigit((*c)) != 0)
		// 		{
		// 			return false;
		// 		}
		// 	}
		// 	return true;
		// };

		std::vector<std::string> PostFixEquations;
		buffer = (*FunctionString.begin());
		bool IsFloat = CheckFloat(buffer);

		// std::cout << FunctionString << std::endl;
		buffer.clear();

		// puts multiplication symbols where they should be
		for (auto c = FunctionString.cbegin(); c != FunctionString.end(); c++)
		{
			buffer += (*c);
			if (std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c)) != SpecialCharacters.end())
			{
				if ((*c) != '(')
				{
					buffer.clear();
					continue;
				}
				if (c != FunctionString.begin() && CheckFloat(std::string(1, *(c - 1))) == false)
				{
					buffer.clear();
					continue;
				}
			}
			bool IsFloatNew = CheckFloat(buffer);
			bool change = !(IsFloat == IsFloatNew);
			if (!change)
			{
				continue;
			}

			if (IsFloat == true && buffer.length() != 1)
			{
				buffer = (*c);
				c = FunctionString.insert(c, '*');
			}

			IsFloat = !IsFloat;
		}

		// std::cout << FunctionString << std::endl;
		buffer.clear();

		// looks for internal functions i.e sin(2exp(x))
		bool InternalFunction = false;
		int BracketDepth = 0;
		int FuncNum = FuncStartNum;
		std::pair<std::string, std::string> funcstring;
		std::vector<std::shared_ptr<Function>> FuncVec;
		std::string FunctionString2 = "";
		for (auto c = FunctionString.begin(); c != FunctionString.end(); c++)
		{
			buffer += (*c);
			FunctionString2 += (*c);

			if (std::find(MathematicalFunctionsList.begin(), MathematicalFunctionsList.end(), buffer) != MathematicalFunctionsList.end() &&
				(c + 1) != FunctionString.end() && (*(c + 1)) == '(')
			{
				InternalFunction = true;
				funcstring.first = buffer;
				buffer.clear();
				continue;
			}

			if (std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c)) != SpecialCharacters.end())
			{
				if (InternalFunction == false)
				{
					buffer.clear();
				}
			}

			if ((*c) == '(')
			{
				BracketDepth++;
			}
			else if ((*c) == ')')
			{
				BracketDepth--;
			}

			if (BracketDepth == 0 && InternalFunction)
			{
				buffer.erase(buffer.begin());
				buffer.erase(buffer.end() - 1);
				funcstring.second = buffer;
				buffer.clear();
				for (unsigned int i = 0; i < (funcstring.first.length() + funcstring.second.length() + 2); i++)
				{
					FunctionString2.erase(FunctionString2.end() - 1);
				}
				FunctionString2 += "f_" + std::to_string(FuncNum);
				FuncNum++;
				InternalFunction = false;
				auto func = FunctionParser(funcstring.first, funcstring.second, FuncNum);
				funcstring.first.clear();
				funcstring.second.clear();
				FuncVec.push_back(func);
				// call the parser on FunctionString2
			}
		}

		// std::cout << FunctionString2 << std::endl;

		// complex number finding
		bool StartReading = false;
		bool Complete = false;
		buffer.clear();
		std::vector<std::pair<std::string, std::string>> ComplexNumbers = {};
		std::vector<std::vector<std::string>> ComplexNumbersVaribles = {};
		std::string::iterator start = FunctionString2.end();
		std::string FunctionString3;

		auto GetPostFixForm = [&SpecialCharacters, &buffer, &prec](std::string Infix)
		{
			buffer.clear();
			std::stack<char> st;
			std::string PostFix2 = "";

			for (auto c = Infix.begin(); c != Infix.end(); c++)
			{
				buffer += (*c);
				auto it3 = std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c));
				if (it3 == SpecialCharacters.end())
				{
					PostFix2 += (*c);
				}
				else if ((*it3) == '(')
				{
					buffer.clear();
					st.push((*c));
				}
				else if ((*it3) == ')')
				{
					buffer.clear();
					while (st.top() != '(')
					{
						PostFix2 += st.top();
						st.pop();
					}
					st.pop();
				}

				else
				{
					buffer.clear();
					while ((!st.empty() && prec(*c) < prec(st.top())) || (!st.empty() && prec(*c) == prec(st.top())))
					{
						PostFix2 += st.top();
						st.pop();
					}
					// std::cout << PostFix2 << std::endl;
					st.push((*c));
				}
			}

			while (!st.empty())
			{
				PostFix2 += st.top();
				st.pop();
			}
			buffer.clear();

			return PostFix2;
		};

		auto FindVarsInValueString = [&SpecialCharacters, &FunctionString3](std::string var)
		{
			std::string InternalBuffer = "";
			std::vector<std::string> ValueVariables;
			bool FoundVar = false;
			for (auto c = var.begin(); c != var.end(); c++)
			{
				if (std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c)) != SpecialCharacters.end() && FoundVar == false)
				{
					continue;
				}

				if ((std::isdigit((*c)) == 0 && FoundVar == false) && ((*c) != '.' && (*c) != ','))
				{
					FoundVar = true;
					InternalBuffer = "";
					InternalBuffer += (*c);
				}
				else if (FoundVar)
				{
					std::string temp = InternalBuffer + (*c);
					auto it3 = FunctionString3.find(temp);
					if (it3 == std::string::npos)
					{
						ValueVariables.push_back(InternalBuffer);
						InternalBuffer.clear();
						FoundVar = false;
						c--;
						continue;
					}
					InternalBuffer += (*c);
				}
			}

			return ValueVariables;
		};

		for (auto c = FunctionString2.begin(); c != FunctionString2.end(); c++)
		{
			std::string temp = buffer + (*c);
			if (temp.compare("Complex(") == 0 || temp.compare("complex(") == 0)
			{
				StartReading = true;
				buffer.clear();
				FunctionString3 += (*c);
				Complete = false;
				start = c - 7;
				continue;
			}
			else if (StartReading && (*c) == ')')
			{
				StartReading = false;
				Complete = true;
			}
			else if (std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*(c))) != SpecialCharacters.end() && !StartReading)
			{
				buffer.clear();
				FunctionString3 += (*c);
				continue;
			}

			if (Complete)
			{
				std::string SecondaryBuffer = "";
				std::pair<std::string, std::string> Value;
				std::vector<char> chr = {'+', '-'};
				for (auto a = buffer.begin(); a != buffer.end(); a++)
				{
					if (std::find(chr.begin(), chr.end(), (*a)) != chr.end())
					{
						auto it1 = std::find(SecondaryBuffer.begin(), SecondaryBuffer.end(), 'i');
						auto it2 = std::find(SecondaryBuffer.begin(), SecondaryBuffer.end(), 'j');
						bool IorJ = (it1 != SecondaryBuffer.end() || it2 != SecondaryBuffer.end());
						if (IorJ)
						{
							if (std::isdigit(*(it1 - 1)) == 0)
							{
								(it1 == SecondaryBuffer.end()) ? SecondaryBuffer.replace(it2, it2 + 1, "1") : SecondaryBuffer.replace(it1, it1 + 1, "1");
							}
							else
							{
								(it1 == SecondaryBuffer.end()) ? SecondaryBuffer.replace(it2, it2 + 1, "") : SecondaryBuffer.replace(it1, it1 + 1, "");
							}

							Value.second += SecondaryBuffer + (*a);
						}
						else
						{
							Value.first += SecondaryBuffer + (*a);
						}
						SecondaryBuffer.clear();
					}
					else
					{
						SecondaryBuffer += (*a);
					}
				}

				if (SecondaryBuffer.size() != 0)
				{
					auto it1 = std::find(SecondaryBuffer.begin(), SecondaryBuffer.end(), 'i');
					auto it2 = std::find(SecondaryBuffer.begin(), SecondaryBuffer.end(), 'j');
					bool IorJ = (it1 != SecondaryBuffer.end() || it2 != SecondaryBuffer.end());
					if (IorJ)
					{
						if (std::isdigit(*(it1 - 1)) == 0)
						{
							(it1 == SecondaryBuffer.end()) ? SecondaryBuffer.replace(it2, it2 + 1, "1") : SecondaryBuffer.replace(it1, it1 + 1, "1");
						}
						else
						{
							(it1 == SecondaryBuffer.end()) ? SecondaryBuffer.replace(it2, it2 + 1, "") : SecondaryBuffer.replace(it1, it1 + 1, "");
						}

						Value.second += SecondaryBuffer;
					}
					else
					{
						Value.first += SecondaryBuffer;
					}
					SecondaryBuffer.clear();
				}

				if (std::find(chr.begin(), chr.end(), (*(Value.first.end() - 1))) != chr.end())
				{
					if (*(Value.first.end() - 1) == '-')
					{
						Value.first.replace(Value.first.end() - 1, Value.first.end(), "");
						Value.second.insert(Value.second.begin(), '-');
					}
					else
					{
						Value.first.replace(Value.first.end() - 1, Value.first.end(), "");
					}
				}

				if (Value.first == "")
					Value.first = "0";
				if (Value.second == "")
					Value.second = "0";

				Value.first = GetPostFixForm(Value.first);
				Value.second = GetPostFixForm(Value.second);

				ComplexNumbers.push_back(Value);

				ComplexNumbersVaribles.push_back(FindVarsInValueString(Value.first));
				std::vector<std::string> temp = FindVarsInValueString(Value.second);
				for (unsigned int i = 0; i < temp.size(); i++)
					ComplexNumbersVaribles[ComplexNumbers.size() - 1].push_back(temp[i]);

				for (int i = 0; i < (c - start); i++)
				{
					FunctionString3.erase(FunctionString3.end() - 1);
				}
				FunctionString3 += "z_" + std::to_string(ComplexNumbers.size() - 1 + s_ComplexNum);
				s_ComplexNum++;
				Complete = false;
				continue;
			}
			buffer += (*c);
			FunctionString3 += (*c);
		}

		// std::cout << FunctionString3 << std::endl;

		std::string PostFix = GetPostFixForm(FunctionString3);
		// std::cout << PostFix << std::endl;
		std::vector<std::string> variables;
		bool variable = false;
		buffer.clear();
		for (auto c = PostFix.begin(); c != PostFix.end(); c++)
		{
			if ((std::isdigit((*c)) || std::find(IgnoreCharacters.begin(), IgnoreCharacters.end(), (*c)) != IgnoreCharacters.end()) && variable == false)
			{
				continue;
			}
			if (std::isdigit((*c)) == 0 && variable == false && std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c)) == SpecialCharacters.end())
			{
				variable = true;
				buffer += (*c);
			}
			else if (variable && std::find(SpecialCharacters.begin(), SpecialCharacters.end(), (*c)) == SpecialCharacters.end())
			{
				std::string temp = buffer + (*c);
				auto it3 = FunctionString3.find(temp);
				if (it3 == std::string::npos)
				{
					variables.push_back(buffer);
					buffer.clear();
					variable = false;
					c--;
					continue;
				}
				buffer += (*c);
			}
			else if (variable)
			{
				auto it3 = FunctionString3.find(buffer);
				if (it3 != std::string::npos)
				{
					variables.push_back(buffer);
					buffer.clear();
					variable = false;
					c--;
					continue;
				}
				buffer += (*c);
			}
		}
		if (variable && !buffer.empty() && FunctionString3.find(buffer) != std::string::npos)
		{
			variables.push_back(buffer);
		}

		// auto ReturnType = [&s_MathmaticalFunctions](std::string MathematicalFunction) {
		// 	std::vector<std::string> vd;
		// 	std::vector<std::string> cdd;
		// 	for(auto a = s_MathmaticalFunctions.cbegin(); a != s_MathmaticalFunctions.end(); a++)
		// 	{
		// 		if(a->second == 0)
		// 		{
		// 			vd.push_back(a->first);
		// 			continue;
		// 		}
		// 		cdd.push_back(a->first);
		// 	}

		// 	if(std::find(vd.begin(), vd.end(), MathematicalFunction) != vd.end()) {
		// 		return Function::VarType::d;}
		// 	else if(std::find(cdd.begin(), cdd.end(), MathematicalFunction) != cdd.end()){
		// 		return Function::VarType::z;}

		// 	throw std::runtime_error("Unknown mathematical function: " + MathematicalFunction);
		// };

		std::vector<Function::VariableDefinition> vardef;
		for (auto v : variables)
		{
			if (v.find("f_") != std::string::npos)
			{
				auto v2 = v;
				v2.erase(v2.begin());
				v2.erase(v2.begin());

				int FuncNum2 = std::stoi(v2) - FuncStartNum;

				vardef.push_back({v, Function::VarType::f, FuncVec[FuncNum2]->GetFunctionString(), FuncVec[FuncNum2]->GetVariable(), FuncVec[FuncNum2]});
			}
			else if (v.find("z_") != std::string::npos)
			{
				auto v2 = v;
				v2.erase(v2.begin());
				v2.erase(v2.begin());

				int zNum = std::stoi(v2) - (s_ComplexNum - ComplexNumbers.size());

				vardef.push_back({v, Function::VarType::z, ComplexNumbers[zNum].first + "#" + ComplexNumbers[zNum].second, ComplexNumbersVaribles[zNum], nullptr});
			}
			else
			{
				vardef.push_back({v, Function::VarType::d, "", {v}, nullptr});
			}
		}

		std::string FunctionName = "";
		std::vector<int> removed = {};
		for (auto c = func.cbegin(); c != func.cend(); c++)
		{
			if ((*c) == '+' || (*c) == '-')
			{
				continue;
			}
			else
			{
				FunctionName += (*c);
				removed.push_back(c - func.begin());
			}
		}
		int r = 0;
		for (unsigned int i = 0; i < removed.size(); i++)
		{
			func.erase(func.begin() + removed[i] - r);
			r++;
		}

		std::shared_ptr<Function> _func = nullptr;
		if (FunctionName.compare("sin") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::sin, Function::ReturnType::d, FunctionName, vardef);
		}
		else if (FunctionName.compare("cos") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::cos, Function::ReturnType::d, FunctionName, vardef);
		}
		else if (FunctionName.compare("scalar") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::scalar, Function::ReturnType::d, FunctionName, vardef);
		}
		else if (FunctionName.compare("scalarcx") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::scalarcx, Function::ReturnType::cd, FunctionName, vardef);
		}
		else if (FunctionName.compare("exp") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::exp, Function::ReturnType::d, FunctionName, vardef);
		}
		else if (FunctionName.compare("expcx") == 0)
		{
			_func = std::make_shared<Function>(MathematicalFunctions::expcx, Function::ReturnType::cd, FunctionName, vardef);
		}
		else
		{
			return nullptr;
		}

		_func->SetFunctionString(FunctionName + "(" + var + ")");
		_func->SetPostFix(PostFix);

		return _func;
	}

	namespace MathematicalFunctions
	{
		void *sin(void *value) // double
		{
			std::complex<double> val2 = *(std::complex<double> *)value;
			double val = val2.real();
			double *_val = new double;
			*_val = std::sin(val);
			return (void *)_val;
		}

		void *cos(void *value) // double
		{
			double *_val = new double;
			std::complex<double> val2 = *(std::complex<double> *)value;
			double val = val2.real();
			*_val = std::cos(val);
			return (void *)_val;
		}

		void *scalar(void *value) // scalar not scaler
		{
			double *_val = new double;
			std::complex<double> val2 = *(std::complex<double> *)value;
			*_val = val2.real();
			return (void *)_val;
		}

		void *scalarcx(void *value)
		{
			std::complex<double> *_val = new std::complex<double>;
			std::complex<double> val = *(std::complex<double> *)value;
			*_val = val;
			return (void *)_val;
		}

		void *exp(void *value)
		{
			double *_val = new double;
			std::complex<double> val2 = *(std::complex<double> *)value;
			double val = val2.real();
			*_val = std::exp(val);
			return (void *)_val;
		}

		void *expcx(void *value)
		{
			std::complex<double> *_val = new std::complex<double>;
			std::complex<double> val = *(std::complex<double> *)value;
			*_val = std::exp(val);
			return (void *)_val;
		}
	}
}
