# IPN-Sistema_analisis_estadistico_datos
Developed for educators and researchers at IPN, this web application is designed to provide a straightforward approach to performing statistical analysis from CSV files. It features tools for data cleaning, transformation, analysis, and report generation, with the added functionality of exporting reports in .docx format. Built with Django.

## Tools
The app was developed using Django as the primary framework. Volt template and Bootstrap 5 were used to speed up the design process and focus on functionality. Essential libraries were integrated to ensure optimal operation and ease of development. Dependency details are specified in **package.json** and **requirements.txt**. Key libraries that support data cleaning, transformation, analysis, and visualization include pandas, numpy, matplotlib, scikit-learn, and scipy.

## System Functions:
### Data Cleaning:
- Column removal
- Text normalization
- Handling missing values
- Outlier processing
- Data filtering based on user-selected conditions

### Data Transformation:
- Handling categorical variables
- Data standardization

### Analysis:
#### Divided into descriptive, inferential, correlation, and machine learning:
- Frequencies
- Measures of central tendency
- Variability
- Hypothesis testing
- Confidence intervals
- Linear regression
- Logistic regression
- Pearson correlation
- Spearman correlation
- K-means clustering
- Binary decision tree
- Multilayer perceptron for classification

## Project Structure
The project utilizes a classic Django MVC (Model-View-Controller) structure. It is organized into different applications that separate the main functions of the system. Within each application, you will find:
- **Templates:** Inheritance is used to avoid redundancies.

- **Views:** Views in the system are responsible for receiving data, processing it through specific functions, and sending responses and data to other screens.

- **Utils:** An additional folder in each application to extend the functionalities of views, especially for data handling and analysis, to avoid overloading the `views.py` files.

At the root of the Django project, there is a `utils` folder containing system-wide useful functions and JS scripts that facilitate the creation of reports in `.docx` format.

## Interface
The app interface consists of two main screens: the loading screen and the main menu. On the loading screen, .csv files from the user are received. If the file is valid, it moves to the main menu.

![imagen menu carga](https://github.com/user-attachments/assets/beb2b3ac-2586-43b9-bb2c-60c9aef55255)

The main menu displays the uploaded data and offers various options through dropdown menus on the sidebar for accessing specific functions. When choosing an operation, like outlier processing, a menu appears allowing the selection of variables and data treatment. 

After data modification, changes such as deleted columns are reported to the user via a green modal in the main menu; a red modal will appear if there is an error when attempting to modify the data. Additionally, the system maintains data versions, allowing a return to previous .csv file versions.

![imagen resultado de eliminar columna](https://github.com/user-attachments/assets/ab85f9a3-62a3-43e7-b165-906dcc366df4)

## Analysis and Reports
For analysis, a similar interface is used for selecting variables and attributes, with some analyses requiring data to be cleaned or transformed beforehand. Once selected and by clicking on **Continue**, the app takes the user to a reports screen where the results are displayed, organized in tables and charts.

![imagen resultado arbol 1](https://github.com/user-attachments/assets/f0dd9a8b-8267-422e-bfb7-d63ce17ec4c1)
![imagen resultado arbol 2](https://github.com/user-attachments/assets/89b45ebd-fe1b-4211-b043-2d4e91927c86)

## Report and CSV exports
Reports can be exported in `.docx` format and modified csv files are available for download.

![imagen reporte exportado, word](https://github.com/user-attachments/assets/6bb0369e-2f67-4efd-b69a-1a4193f37465)

## Tooltips
The application also features a series of tooltips and modals that guide the user through the application's functions and provide brief explanations of the report results.

![imagen modal apoyo, seleccion atributos](https://github.com/user-attachments/assets/aa8c2421-1795-4d70-b377-76cf444cdd56)
![imagen ejemplo tooltip reporte](https://github.com/user-attachments/assets/90570b09-d563-4818-afbc-add739a63c8e)

## Areas for Improvement:
- **Column Management:** Currently, an error allows users to delete all columns in the CSV, which can cause the application to crash as it assumes there will always be a CSV to proceed to the next screen.

- **Temporary Storage:** The `temp_files` folder is used to store users' CSV files and images generated for reports. Implementing a database or periodically clearing the folder would be better.

- **Unused Files:** Some files from the Volt template are not used in the project and could be removed to optimize space and loading.

- **Code Redundancy:** There is redundancy in the code and HTML templates. Implementing template inheritance could unify and reduce the number of lines of code.

## Credits
- Aarón Cano
- Alejandro Chávez
- Jesús Valencia
